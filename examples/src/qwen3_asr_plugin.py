"""glide plugin: Qwen3-ASR (speech-LLM) support.

Qwen3-ASR is a custom architecture shipped in the ``qwen_asr`` package (not a
built-in transformers model), so this plugin:

1. Registers ``Qwen3ASRConfig`` / ``Qwen3ASRForConditionalGeneration`` /
   ``Qwen3ASRProcessor`` with the HF Auto classes so glide's generic loader can
   load it via ``model.auto_class: AutoModel``.
2. Patches the model's outer ``forward`` to delegate to ``thinker.forward``
   (matching Qwen3-ASR's official finetuning recipe).
3. Registers a ``qwen3_asr`` data collator that mirrors the official recipe's
   *prefix masking*: it runs the processor twice (once on ``prefix + target`` and
   once on the ``prefix`` alone) and masks the prefix length, which is robust to
   audio-token expansion.

Use it from a config::

    plugins: ["examples/src/qwen3_asr_plugin.py"]
    model: { name: Qwen/Qwen3-ASR-1.7B, auto_class: AutoModel }
    template: { collator: qwen3_asr, system_prompt: "Transcribe the audio." }
    data: { audio_field: audio, response_field: text }

Requires the transformers==4.57.6-pinned environment (``.venv-qwenasr``); the
modeling code is incompatible with transformers 5.x.
"""

from dataclasses import dataclass
from typing import Any

import torch

from glide.data.audio import load_audio
from glide.data.packing import block_diagonal_causal_mask, cu_seqlens, packed_position_ids
from glide.registry import collators


def _register_qwen3_asr() -> None:
    """Register config/model/processor with the HF Auto classes (idempotent)."""
    import importlib

    from transformers import AutoConfig, AutoModel, AutoProcessor

    # Dynamic import: qwen_asr is an optional dependency installed only in the
    # Qwen3-ASR environment, so don't hard-import it at module scope.
    backend = importlib.import_module("qwen_asr.core.transformers_backend")
    cfg_cls = backend.Qwen3ASRConfig

    AutoConfig.register("qwen3_asr", cfg_cls, exist_ok=True)
    AutoModel.register(cfg_cls, backend.Qwen3ASRForConditionalGeneration, exist_ok=True)
    try:
        AutoProcessor.register(cfg_cls, backend.Qwen3ASRProcessor, exist_ok=True)
    except (ValueError, KeyError):
        pass  # already registered

    _patch_forward(backend.Qwen3ASRForConditionalGeneration)


def _patch_forward(cls) -> None:
    """Make the outer model forward delegate to ``thinker.forward`` (once)."""
    if getattr(cls, "_glide_forward_patched", False):
        return

    def forward(self, input_ids=None, attention_mask=None, input_features=None,
                feature_attention_mask=None, labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._glide_forward_patched = True


@dataclass
class Qwen3ASRCollator:
    """Prefix-masked collator for Qwen3-ASR SFT (mirrors the official recipe)."""

    processor: Any
    audio_field: str = "audio"
    target_field: str = "text"
    system_prompt: str = ""
    sampling_rate: int = 16000

    def _prefix_messages(self):
        return [
            {"role": "system", "content": self.system_prompt or ""},
            {"role": "user", "content": [{"type": "audio", "audio": None}]},
        ]

    def __call__(self, features: list[dict]) -> dict:
        tokenizer = self.processor.tokenizer
        eos = tokenizer.eos_token or ""
        prefix_text = self.processor.apply_chat_template(
            [self._prefix_messages()], add_generation_prompt=True, tokenize=False
        )[0]

        audios = [load_audio(f[self.audio_field], self.sampling_rate) for f in features]
        targets = [f[self.target_field] for f in features]
        prefix_texts = [prefix_text] * len(features)
        full_texts = [prefix_text + t + eos for t in targets]

        full = self.processor(text=full_texts, audio=audios, return_tensors="pt",
                              padding=True, truncation=False)
        prefix = self.processor(text=prefix_texts, audio=audios, return_tensors="pt",
                                padding=True, truncation=False)

        prefix_lens = prefix["attention_mask"].sum(dim=1).tolist()
        labels = full["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, : int(pl)] = -100
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full["labels"] = labels
        return full


@dataclass
class Qwen3ASRPackedCollator:
    """Sequence-*packing* collator for Qwen3-ASR SFT (SDPA or FA2-varlen).

    Concatenates the batch's examples into a single unpadded row instead of
    padding to the longest. Feasible because (1) the audio tower already accepts
    multiple audios, (2) ``masked_scatter`` fills the N audio placeholder spans in
    order, and (3) Qwen3-ASR's mRoPE is plain sequential positions, so packing-aware
    ``position_ids`` are just per-example ``arange`` resets.

    Cross-example attention must be blocked. Two backends:

    * ``attn_mode="sdpa"`` -- a dense (T,T) block-diagonal boolean mask. Correct
      but attention is O(T_pack^2) over the whole pack (SDPA doesn't exploit the
      block structure).
    * ``attn_mode="fa2"`` -- pass ``cu_seq_lens_{q,k}`` + ``max_length_{q,k}`` so
      flash-attention-2 runs the *varlen* kernel: attention is O(sum L_i^2) AND
      there is zero padding. This is the most compute-efficient option; it
      requires ``model.attn_implementation='flash_attention_2'``.

    Packs are built by CONCATENATING per-example processor outputs (exact; avoids
    BPE merges drifting the boundaries of a single combined-text tokenization).
    """

    processor: Any
    audio_field: str = "audio"
    target_field: str = "text"
    system_prompt: str = ""
    sampling_rate: int = 16000
    attn_mode: str = "sdpa"  # "sdpa" | "fa2"

    def _prefix_text(self):
        msgs = [
            {"role": "system", "content": self.system_prompt or ""},
            {"role": "user", "content": [{"type": "audio", "audio": None}]},
        ]
        return self.processor.apply_chat_template(
            [msgs], add_generation_prompt=True, tokenize=False
        )[0]

    def __call__(self, features: list[dict]) -> dict:

        tok = self.processor.tokenizer
        eos = tok.eos_token or ""
        prefix = self._prefix_text()
        audios = [load_audio(f[self.audio_field], self.sampling_rate) for f in features]
        targets = [f[self.target_field] for f in features]

        ids_list, feat_list, fmask_list, e_lens, p_lens = [], [], [], [], []
        for tgt, aud in zip(targets, audios):
            full = self.processor(text=[prefix + tgt + eos], audio=[aud],
                                  return_tensors="pt", truncation=False)
            p = self.processor(text=[prefix], audio=[aud],
                               return_tensors="pt", truncation=False)["input_ids"].shape[1]
            ids_list.append(full["input_ids"][0])
            feat_list.append(full["input_features"])           # (1, D, T_i)
            fmask_list.append(full["feature_attention_mask"])  # (1, T_i)
            e_lens.append(full["input_ids"].shape[1])
            p_lens.append(p)

        total = sum(e_lens)
        packed = {"input_ids": torch.cat(ids_list).unsqueeze(0)}  # (1, total)

        # Stack per-example audio features (pad the time dim to the pack max).
        max_t = max(f.shape[-1] for f in feat_list)
        D = feat_list[0].shape[1]
        feats = feat_list[0].new_zeros(len(feat_list), D, max_t)
        fmask = fmask_list[0].new_zeros(len(fmask_list), max_t)
        for i, (f, m) in enumerate(zip(feat_list, fmask_list)):
            feats[i, :, : f.shape[-1]] = f[0]
            fmask[i, : m.shape[-1]] = m[0]
        packed["input_features"] = feats
        packed["feature_attention_mask"] = fmask

        # mRoPE position_ids reset to 0 at every example boundary (sequential rope).
        pos = packed_position_ids(e_lens).view(1, -1)
        packed["position_ids"] = pos.unsqueeze(0).expand(3, 1, total).clone()

        if self.attn_mode == "fa2":
            # FA2 varlen: hand the kernel the segment boundaries; flash-attn does
            # block-diagonal causal attention within each segment. No dense mask.
            cu, max_len = cu_seqlens(e_lens)
            packed["cu_seq_lens_q"] = cu
            packed["cu_seq_lens_k"] = cu
            packed["max_length_q"] = max_len
            packed["max_length_k"] = max_len
            # Do NOT add an "attention_mask" key (not even None): the model's FA2
            # varlen path uses cu_seqlens, and TRL's loss code does
            # `entropy * inputs["attention_mask"]` whenever the key is present,
            # which would be `Tensor * None`. Omitting it routes TRL to position_ids.
        else:
            # SDPA: dense (1,1,T,T) block-diagonal causal bool mask (True=attend).
            packed["attention_mask"] = block_diagonal_causal_mask(e_lens)

        # labels: keep targets, mask each example's prefix span.
        labels = packed["input_ids"].clone()
        start = 0
        for e, p in zip(e_lens, p_lens):
            labels[0, start : start + p] = -100
            start += e
        pad_id = tok.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        packed["labels"] = labels
        return packed


@collators.register("qwen3_asr_packed", exist_ok=True)
def build_qwen3_asr_packed_collator(config, processor):
    """Builder for the packing collator (``template.collator: qwen3_asr_packed``).

    Correctness depends on a dense block-diagonal attention mask, which only the
    ``sdpa``/``eager`` backends honor. ``flash_attention_2`` silently ignores dense
    masks (it uses ``is_causal``/``cu_seqlens``), which would leak attention across
    packed examples and train incorrectly -- so we refuse to build under FA2.
    """
    _register_qwen3_asr()
    if config.model.attn_implementation not in ("sdpa", "eager"):
        raise ValueError(
            "qwen3_asr_packed requires model.attn_implementation 'sdpa' or 'eager': "
            f"got {config.model.attn_implementation!r}. A dense block-diagonal mask "
            "is ignored by flash_attention_2, which would silently leak attention "
            "across packed examples. Use sdpa/eager, or implement a cu_seqlens "
            "(varlen) packing path for FA2."
        )
    return Qwen3ASRPackedCollator(
        processor=processor,
        audio_field=config.data.audio_field,
        target_field=config.data.response_field,
        system_prompt=config.template.system_prompt or "",
        sampling_rate=config.speech.sample_rate,
        attn_mode="sdpa",
    )


@collators.register("qwen3_asr_packed_fa2", exist_ok=True)
def build_qwen3_asr_packed_fa2_collator(config, processor):
    """Packing collator using FlashAttention-2 varlen (``cu_seq_lens``).

    The most compute-efficient option: per-segment (O(sum L_i^2)) attention with
    zero padding. Requires ``model.attn_implementation='flash_attention_2'`` (the
    SDPA dense-mask path would be O(T_pack^2)).
    """
    _register_qwen3_asr()
    if config.model.attn_implementation != "flash_attention_2":
        raise ValueError(
            "qwen3_asr_packed_fa2 requires model.attn_implementation="
            f"'flash_attention_2': got {config.model.attn_implementation!r}."
        )
    return Qwen3ASRPackedCollator(
        processor=processor,
        audio_field=config.data.audio_field,
        target_field=config.data.response_field,
        system_prompt=config.template.system_prompt or "",
        sampling_rate=config.speech.sample_rate,
        attn_mode="fa2",
    )


@collators.register("qwen3_asr", exist_ok=True)
def build_qwen3_asr_collator(config, processor):
    """Builder used by glide's SFT path (``template.collator: qwen3_asr``)."""
    _register_qwen3_asr()
    return Qwen3ASRCollator(
        processor=processor,
        audio_field=config.data.audio_field,
        target_field=config.data.response_field,
        system_prompt=config.template.system_prompt or "",
        sampling_rate=config.speech.sample_rate,
    )


# Register at import so the Auto classes are ready before the model loads.
_register_qwen3_asr()
