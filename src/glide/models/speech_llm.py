"""Composable Speech-LLM: ``encoder -> projector -> text LLM``.

Assembles a speech model from three independently-chosen parts:

* an audio **encoder** (Whisper / WavLM / XLS-R / Qwen3-ASR AuT / Qwen-Omni AuT),
* a **projector** (from-scratch MLP+GeLU, or a pretrained Qwen projector),
* a text **LLM** (Qwen3-1.7B, Llama-3.2-1B/3B, Gemma-3-1B/4B, ...).

At ``forward``/``generate`` the projected audio frames are *spliced into* the LLM
input embeddings at each sample's ``<audio>`` marker, so the data collator only
needs to place a single marker token (no need to pre-compute audio-token counts).
Audio positions are excluded from the loss.
"""

import torch
import torch.nn as nn

__all__ = ["SpeechLLM", "build_speech_llm"]


class SpeechLLM(nn.Module):
    """Encoder + projector + LLM with runtime audio-embedding splicing."""

    def __init__(self, encoder, projector, llm, audio_token_id: int, downsample: int = 1):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.llm = llm
        self.audio_token_id = audio_token_id
        self.downsample = max(1, downsample)
        self.config = llm.config  # so HF Trainer / generation utilities see a config

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate gradient checkpointing to the LLM (and the trainable encoder)."""
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable(**kwargs)
        enc = getattr(self.encoder, "model", None)
        if enc is not None and hasattr(enc, "gradient_checkpointing_enable"):
            try:
                enc.gradient_checkpointing_enable(**kwargs)
            except Exception:
                pass  # some encoders don't support it; the LLM still benefits

    def gradient_checkpointing_disable(self):
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            self.llm.gradient_checkpointing_disable()
        enc = getattr(self.encoder, "model", None)
        if enc is not None and hasattr(enc, "gradient_checkpointing_disable"):
            try:
                enc.gradient_checkpointing_disable()
            except Exception:
                pass

    def _encode_audio(self, audio_inputs: dict):
        """Return ``(audio_embeds (B,Ta,H), audio_lengths (B,))`` after projection."""
        # Cast float audio features to the encoder dtype (bf16): the feature extractor
        # emits fp32, and the FA2 audio tower requires fp16/bf16. Training survived via
        # the Trainer's autocast; this makes it backend-agnostic (incl. the eval loop).
        enc_dtype = next(self.encoder.parameters()).dtype
        audio_inputs = {k: (v.to(enc_dtype) if torch.is_floating_point(v) else v)
                        for k, v in audio_inputs.items()}
        hidden, frame_mask = self.encoder(**audio_inputs)
        embeds = self.projector(hidden)
        lengths = frame_mask.sum(-1) if frame_mask is not None else \
            torch.full((hidden.shape[0],), hidden.shape[1], device=hidden.device)
        lengths = (lengths // self.downsample).clamp(max=embeds.shape[1])
        return embeds, lengths

    def _splice(self, input_ids, attention_mask, labels, audio_embeds, audio_lengths):
        """Replace each sample's single ``<audio>`` marker with its audio frames."""
        device = input_ids.device
        embed_layer = self.get_input_embeddings()
        rows_embeds, rows_labels, rows_len = [], [], []

        for b in range(input_ids.shape[0]):
            ids = input_ids[b]
            keep = attention_mask[b].bool() if attention_mask is not None else torch.ones_like(ids, dtype=torch.bool)
            ids_valid = ids[keep]
            lab_valid = labels[b][keep] if labels is not None else None

            marker = (ids_valid == self.audio_token_id).nonzero(as_tuple=True)[0]
            txt_embeds = embed_layer(ids_valid)
            if len(marker) == 0:
                rows_embeds.append(txt_embeds)
                rows_labels.append(lab_valid if lab_valid is not None else None)
                rows_len.append(txt_embeds.shape[0])
                continue

            pos = int(marker[0])
            a = audio_embeds[b, : int(audio_lengths[b])]  # (La, H)
            new_embed = torch.cat([txt_embeds[:pos], a, txt_embeds[pos + 1:]], dim=0)
            rows_embeds.append(new_embed)
            if lab_valid is not None:
                audio_lab = torch.full((a.shape[0],), -100, dtype=lab_valid.dtype, device=device)
                rows_labels.append(torch.cat([lab_valid[:pos], audio_lab, lab_valid[pos + 1:]]))
            rows_len.append(new_embed.shape[0])

        max_len = max(rows_len)
        h = rows_embeds[0].shape[-1]
        bsz = len(rows_embeds)
        out_embeds = rows_embeds[0].new_zeros(bsz, max_len, h)
        out_mask = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
        out_labels = None
        if labels is not None:
            out_labels = torch.full((bsz, max_len), -100, dtype=labels.dtype, device=device)
        for b, (e, n) in enumerate(zip(rows_embeds, rows_len)):
            out_embeds[b, :n] = e
            out_mask[b, :n] = 1
            if out_labels is not None and rows_labels[b] is not None:
                out_labels[b, :n] = rows_labels[b]
        return out_embeds, out_mask, out_labels

    @staticmethod
    def _encoder_inputs(input_features, feature_attention_mask, input_values, audio_attention_mask):
        """Build the encoder's kwargs from the (collision-free) audio inputs."""
        enc: dict = {}
        if input_features is not None:
            enc["input_features"] = input_features
            if feature_attention_mask is not None:
                enc["feature_attention_mask"] = feature_attention_mask
        if input_values is not None:
            enc["input_values"] = input_values
            if audio_attention_mask is not None:
                enc["attention_mask"] = audio_attention_mask  # encoder's own arg name
        return enc

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                input_features=None, feature_attention_mask=None,
                input_values=None, audio_attention_mask=None, **kwargs):
        # Ignore extra Trainer kwargs (use_cache, num_items_in_batch, ...): training
        # uses no cache and the loss is computed by the LLM over the spliced labels.
        enc = self._encoder_inputs(input_features, feature_attention_mask,
                                   input_values, audio_attention_mask)
        audio_embeds, audio_lengths = self._encode_audio(enc)
        inputs_embeds, attn, new_labels = self._splice(
            input_ids, attention_mask, labels, audio_embeds, audio_lengths
        )
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=new_labels)
        # Stash next-token accuracy counts over the spliced labels (the trainer reads
        # these to log train/eval mean_token_accuracy -- the spliced labels only exist
        # here, not in the Trainer's text-length inputs["labels"]).
        if new_labels is not None and getattr(out, "logits", None) is not None:
            with torch.no_grad():
                shift_logits = out.logits[..., :-1, :]
                shift_labels = new_labels[..., 1:]
                mask = shift_labels != -100
                preds = shift_logits.argmax(dim=-1)
                self._tok_correct = ((preds == shift_labels) & mask).sum()
                self._tok_total = mask.sum()
        return out

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None,
                 input_features=None, feature_attention_mask=None,
                 input_values=None, audio_attention_mask=None, **kwargs):
        enc = self._encoder_inputs(input_features, feature_attention_mask,
                                   input_values, audio_attention_mask)
        audio_embeds, audio_lengths = self._encode_audio(enc)
        inputs_embeds, attn, _ = self._splice(input_ids, attention_mask, None,
                                              audio_embeds, audio_lengths)
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attn, **kwargs)


def build_speech_llm(config):
    """Assemble a composed Speech-LLM.

    Returns ``(model, tokenizer, special_token_info)`` -- the tokenizer is returned
    rather than attached to the ``nn.Module`` (whose ``__setattr__`` only accepts
    tensors/modules).
    """
    import transformers

    from ..registry import audio_encoders, projectors
    from .loader import resolve_dtype
    from .special_tokens import apply_special_tokens

    speech = config.speech
    dtype = resolve_dtype(config.model.torch_dtype)
    enc_builder = audio_encoders.get(speech.encoder.name)
    encoder = enc_builder(speech.encoder, speech.sample_rate)
    # The AuT loads in fp32 by default; cast to the model dtype (bf16) so it matches the
    # LLM and the FA2 audio tower gets fp16/bf16 (FA2 rejects fp32) -- needed at eval,
    # where there's no autocast (training survived via the Trainer's autocast).
    encoder = encoder.to(dtype)

    llm = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name,
        dtype=dtype,
        attn_implementation=config.model.attn_implementation,
        trust_remote_code=config.model.trust_remote_code,
    )

    proj_builder = projectors.get(speech.projector.name)
    projector = proj_builder(speech.projector, encoder.output_dim, llm.config.hidden_size)

    # Tokenizer/special tokens come from the LLM; the <audio> marker is required.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_name or config.model.name)
    info = apply_special_tokens(tokenizer, llm, config.special_tokens)
    audio_id = info.audio_token_id
    if audio_id is None:
        raise ValueError(
            "A composed Speech-LLM needs special_tokens.audio_token (the <audio> "
            "marker spliced with audio embeddings)."
        )
    model = SpeechLLM(encoder, projector, llm, audio_token_id=audio_id,
                      downsample=speech.projector.downsample)
    return model, tokenizer, info
