"""Built-in audio encoders for composed Speech-LLMs.

Each entry in :data:`glide.registry.audio_encoders` is a builder
``(AudioEncoderConfig, sample_rate) -> AudioEncoder`` returning an
:class:`~glide.models.plugins_base.AudioEncoder` that exposes:

* ``output_dim``    -- hidden size of the encoder output.
* ``input_kind``    -- ``"input_features"`` (log-mel; Whisper/Qwen) or
  ``"input_values"`` (raw waveform; WavLM/XLS-R).
* ``feature_extractor`` -- the HF feature extractor that turns waveforms into the
  encoder's inputs (used by the data collator).
* ``forward(**inputs) -> (hidden_states, frame_mask)``.

Heavy / optional model code (Qwen3-ASR, Qwen-Omni) is imported lazily inside the
builder so importing this module never requires those packages.
"""

import torch

from ..config.schema import AudioEncoderConfig
from ..registry import audio_encoders
from .plugins_base import AudioEncoder

__all__ = ["WhisperEncoder", "WaveformEncoder"]


# Friendly size aliases for Whisper.
_WHISPER_ALIASES = {
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large",
    "large-v1": "openai/whisper-large",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
}


def _maybe_freeze(module, freeze: bool):
    if freeze:
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()
    return module


class WhisperEncoder(AudioEncoder):
    """Whisper encoder stack (log-mel input)."""

    input_kind = "input_features"

    def __init__(self, name: str, freeze: bool = False, **kwargs):
        super().__init__()
        from transformers import AutoFeatureExtractor, WhisperModel

        repo = _WHISPER_ALIASES.get(name, name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(repo)
        self.model = WhisperModel.from_pretrained(repo, **kwargs).encoder
        self.output_dim = self.model.config.d_model
        _maybe_freeze(self.model, freeze)

    def forward(self, input_features, attention_mask=None, **_):
        out = self.model(input_features=input_features).last_hidden_state
        # Whisper pads to a fixed 30s grid; every output frame is valid.
        mask = torch.ones(out.shape[:2], dtype=torch.long, device=out.device)
        return out, mask


class WaveformEncoder(AudioEncoder):
    """Raw-waveform self-supervised encoder (WavLM / Wav2Vec2 / XLS-R)."""

    input_kind = "input_values"

    def __init__(self, repo: str, arch: str = "auto", freeze: bool = False, **kwargs):
        super().__init__()
        import transformers
        from transformers import AutoFeatureExtractor

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(repo)
        cls = {
            "wavlm": getattr(transformers, "WavLMModel", None),
            "wav2vec2": getattr(transformers, "Wav2Vec2Model", None),
        }.get(arch) or transformers.AutoModel
        self.model = cls.from_pretrained(repo, **kwargs)
        self.output_dim = self.model.config.hidden_size
        _maybe_freeze(self.model, freeze)

    def forward(self, input_values, attention_mask=None, **_):
        out = self.model(input_values=input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        if attention_mask is not None and hasattr(self.model, "_get_feature_vector_attention_mask"):
            mask = self.model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        else:
            mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=hidden.device)
        return hidden, mask


@audio_encoders.register("whisper", exist_ok=True)
def build_whisper(cfg: AudioEncoderConfig, sample_rate: int = 16000):
    return WhisperEncoder(cfg.pretrained or "openai/whisper-large-v3",
                          freeze=cfg.freeze, **cfg.extra_kwargs)


@audio_encoders.register("wavlm", exist_ok=True)
def build_wavlm(cfg: AudioEncoderConfig, sample_rate: int = 16000):
    return WaveformEncoder(cfg.pretrained or "microsoft/wavlm-large",
                           arch="wavlm", freeze=cfg.freeze, **cfg.extra_kwargs)


@audio_encoders.register("xls_r", exist_ok=True)
def build_xls_r(cfg: AudioEncoderConfig, sample_rate: int = 16000):
    return WaveformEncoder(cfg.pretrained or "facebook/wav2vec2-xls-r-300m",
                           arch="wav2vec2", freeze=cfg.freeze, **cfg.extra_kwargs)


@audio_encoders.register("qwen3_asr_aut", exist_ok=True)
def build_qwen3_asr_aut(cfg: AudioEncoderConfig, sample_rate: int = 16000):
    """Qwen3-ASR audio tower (requires the ``qwen_asr`` package).

    Extracts ``.thinker.audio_tower`` from a Qwen3-ASR checkpoint and its processor's
    feature extractor. Input is log-mel ``input_features`` + ``feature_attention_mask``.
    """
    import importlib
    import importlib.util

    from transformers import AutoConfig, AutoModel, AutoProcessor

    # Optional dependency: import dynamically so a static checker / a bare install
    # without the qwen_asr package doesn't choke on this builder. Register the
    # custom arch with the Auto classes so from_pretrained works (self-contained;
    # no separate plugin needed just for the encoder).
    backend = importlib.import_module("qwen_asr.core.transformers_backend")
    AutoConfig.register("qwen3_asr", backend.Qwen3ASRConfig, exist_ok=True)
    AutoModel.register(backend.Qwen3ASRConfig, backend.Qwen3ASRForConditionalGeneration,
                       exist_ok=True)
    # Per-audio output length (post-CNN) -- used to split the varlen output per utterance.
    _out_len_fn = backend.modeling_qwen3_asr._get_feat_extract_output_lengths

    repo = cfg.pretrained or "Qwen/Qwen3-ASR-1.7B"
    full = AutoModel.from_pretrained(repo, **cfg.extra_kwargs)
    tower = full.thinker.audio_tower
    out_dim = int(getattr(tower.config, "output_dim", None)
                  or full.config.get_text_config().hidden_size)

    # The AuT only honors cu_seqlens (true per-utterance block-diagonal attention) under
    # flash_attention_2; under sdpa/eager a concatenated call leaks across utterances. So
    # force FA2 on the tower -> enables the fast leak-free *varlen* path (one call for the
    # whole batch). If FA2 isn't available, fall back to the per-audio loop (leak-free too).
    use_varlen = False
    if importlib.util.find_spec("flash_attn") is not None:
        for _m in [tower, *tower.modules()]:
            _c = getattr(_m, "config", None)
            if _c is not None:
                try:
                    _c._attn_implementation = "flash_attention_2"
                except Exception:
                    pass
            if hasattr(_m, "_attn_implementation"):
                _m._attn_implementation = "flash_attention_2"
        use_varlen = getattr(tower.config, "_attn_implementation", None) == "flash_attention_2"
    print(f"[glide] qwen3_asr_aut: {'FA2 varlen (batched, leak-free)' if use_varlen else 'per-audio loop (sdpa fallback)'}",
          flush=True)

    class _Tower(AudioEncoder):
        input_kind = "input_features"

        use_varlen = False  # set on the instance below

        def __init__(self):
            super().__init__()
            self.model = tower
            self.feature_extractor = AutoProcessor.from_pretrained(repo).feature_extractor
            self.output_dim = out_dim
            self.use_varlen = use_varlen
            _maybe_freeze(self.model, cfg.freeze)

        def _feat_lens(self, input_features, feature_attention_mask):
            n_frames = input_features.shape[-1]
            if feature_attention_mask is not None:
                # WhisperFeatureExtractor's attention_mask is sample-level (e.g. 480000);
                # convert valid samples -> valid mel frames (= samples * n_frames / mask_len).
                mask_len = feature_attention_mask.shape[1]
                valid = feature_attention_mask.sum(dim=1).float()
                return (valid * n_frames / mask_len).round().long().clamp(1, n_frames)
            return torch.full((input_features.shape[0],), n_frames,
                              device=input_features.device, dtype=torch.long)

        @staticmethod
        def _pack(chunks, device):
            lmax = max(c.shape[0] for c in chunks)
            hidden = chunks[0].new_zeros(len(chunks), lmax, chunks[0].shape[-1])
            mask = torch.zeros(len(chunks), lmax, dtype=torch.long, device=device)
            for b, c in enumerate(chunks):
                hidden[b, : c.shape[0]] = c
                mask[b, : c.shape[0]] = 1
            return hidden, mask

        def forward(self, input_features, feature_attention_mask=None, **_):
            feat_lens = self._feat_lens(input_features, feature_attention_mask)
            if self.use_varlen:
                # Varlen (FA2): ONE call over all utterances concatenated. The AuT builds
                # cu_seqlens from feature_lens and FA2 attends block-diagonally per utterance
                # (verified leak-free: audio0 output is unchanged when other audios change).
                feats = torch.cat([input_features[b][:, : int(feat_lens[b])]
                                   for b in range(input_features.shape[0])], dim=1)
                out = self.model(feats, feature_lens=feat_lens)
                flat = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
                chunks = list(flat.split(_out_len_fn(feat_lens).tolist(), dim=0))
                return self._pack(chunks, flat.device)
            # Fallback: per-audio loop (leak-free under any backend, but sequential/slower).
            outs = []
            for b in range(input_features.shape[0]):
                fl = int(feat_lens[b])
                out = self.model(input_features[b][:, :fl], feature_lens=feat_lens[b:b + 1])
                outs.append(out.last_hidden_state if hasattr(out, "last_hidden_state") else out)
            return self._pack(outs, outs[0].device)

    return _Tower()


def _find_audio_tower(module):
    """Locate the ``audio_tower`` submodule regardless of Omni version/nesting."""
    for path in ("audio_tower", "thinker.audio_tower", "model.audio_tower",
                 "thinker.model.audio_tower"):
        obj = module
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    # Fallback: search named submodules for one called 'audio_tower'.
    for name, sub in module.named_modules():
        if name.endswith("audio_tower"):
            return sub
    raise AttributeError("could not locate audio_tower in the Omni model")


@audio_encoders.register("qwen_omni_aut", exist_ok=True)
def build_qwen_omni_aut(cfg: AudioEncoderConfig, sample_rate: int = 16000):
    """Qwen-Omni audio tower — **Qwen2.5-Omni or Qwen3-Omni**, selected by ``pretrained``.

    Version-agnostic: loads via ``AutoModel`` (the right Omni class is chosen from the
    checkpoint config) and locates the ``audio_tower`` submodule. Set
    ``pretrained="Qwen/Qwen2.5-Omni-7B"`` (default) or a Qwen3-Omni id
    (e.g. ``Qwen/Qwen3-Omni-...``).
    """
    from transformers import AutoFeatureExtractor, AutoModel

    repo = cfg.pretrained or "Qwen/Qwen2.5-Omni-7B"
    full = AutoModel.from_pretrained(repo, trust_remote_code=True, **cfg.extra_kwargs)
    tower = _find_audio_tower(full)
    # The tower's self.proj emits config.output_dim (e.g. 3584); config.d_model (1280) is
    # the *internal* width. Prefer output_dim so the projector's in_dim matches reality.
    out_dim = int(getattr(tower.config, "output_dim", None)
                  or getattr(tower.config, "d_model", 0))

    return _build_omni_tower(tower, repo, out_dim, cfg.freeze)


def _build_omni_tower(tower, repo: str, out_dim: int, freeze: bool):
    from transformers import AutoFeatureExtractor

    class _OmniTower(AudioEncoder):
        input_kind = "input_features"

        def __init__(self):
            super().__init__()
            self.model = tower
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(repo)
            self.output_dim = out_dim
            _maybe_freeze(self.model, freeze)

        def _feat_lens(self, input_features, feature_attention_mask):
            n_frames = input_features.shape[-1]
            if feature_attention_mask is not None:
                # Normalize whatever resolution the mask is at (sample- or frame-level)
                # to valid mel frames via the length ratio.
                mask_len = feature_attention_mask.shape[1]
                valid = feature_attention_mask.sum(dim=1).float()
                return (valid * n_frames / mask_len).round().long().clamp(1, n_frames)
            return torch.full((input_features.shape[0],), n_frames,
                              device=input_features.device, dtype=torch.long)

        @staticmethod
        def _pack(chunks, device):
            lmax = max(c.shape[0] for c in chunks)
            hidden = chunks[0].new_zeros(len(chunks), lmax, chunks[0].shape[-1])
            mask = torch.zeros(len(chunks), lmax, dtype=torch.long, device=device)
            for b, c in enumerate(chunks):
                hidden[b, : c.shape[0]] = c
                mask[b, : c.shape[0]] = 1
            return hidden, mask

        def forward(self, input_features, feature_attention_mask=None, **_):
            # The Omni audio tower consumes the *flat varlen* form (num_mel_bins,
            # sum_frames) with an explicit feature_lens (without it,
            # chunk_and_pad_features(..., None, ...) crashes), and returns a flat
            # (sum_pooled_frames, output_dim) tensor -- NOT (B, T, H). Concatenate the
            # valid mel frames across utterances, then split the output back per-utterance
            # into the padded (B, T, H) that SpeechLLM._encode_audio expects.
            feat_lens = self._feat_lens(input_features, feature_attention_mask)
            feats = torch.cat(
                [input_features[b][:, : int(feat_lens[b])] for b in range(input_features.shape[0])],
                dim=1,
            )
            out = self.model(feats, feature_lens=feat_lens)
            flat = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            _, out_lens = self.model._get_feat_extract_output_lengths(feat_lens)
            chunks = list(flat.split([int(x) for x in out_lens], dim=0))
            return self._pack(chunks, flat.device)

    return _OmniTower()
