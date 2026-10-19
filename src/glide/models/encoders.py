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
from ..logging_utils import get_logger
from ..registry import audio_encoders
from .plugins_base import AudioEncoder

_log = get_logger("models")

__all__ = [
    "WhisperEncoder",
    "WaveformEncoder",
    "Qwen3ASRAudioTowerEncoder",
    "QwenOmniAudioTowerEncoder",
]


# Friendly size aliases for Whisper.
_WHISPER_ALIASES = {
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large",
    "large-v1": "openai/whisper-large",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
}


def _freeze_module(module) -> None:
    """Stop gradients through every parameter of ``module`` and put it in eval mode."""
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


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
        if freeze:
            _freeze_module(self.model)

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
        if freeze:
            _freeze_module(self.model)

    def forward(self, input_values, attention_mask=None, **_):
        out = self.model(input_values=input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        if attention_mask is not None and hasattr(self.model, "_get_feature_vector_attention_mask"):
            mask = self.model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        else:
            mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=hidden.device)
        return hidden, mask


@audio_encoders.register("whisper", exist_ok=True)
def build_whisper_encoder(cfg: AudioEncoderConfig, sample_rate: int = 16000):
    return WhisperEncoder(cfg.pretrained or "openai/whisper-large-v3",
                          freeze=cfg.freeze, **cfg.extra_kwargs)


@audio_encoders.register("wavlm", exist_ok=True)
def build_wavlm_encoder(cfg: AudioEncoderConfig, sample_rate: int = 16000):
    return WaveformEncoder(cfg.pretrained or "microsoft/wavlm-large",
                           arch="wavlm", freeze=cfg.freeze, **cfg.extra_kwargs)


@audio_encoders.register("xls_r", exist_ok=True)
def build_xls_r_encoder(cfg: AudioEncoderConfig, sample_rate: int = 16000):
    return WaveformEncoder(cfg.pretrained or "facebook/wav2vec2-xls-r-300m",
                           arch="wav2vec2", freeze=cfg.freeze, **cfg.extra_kwargs)


class _QwenAudioTowerEncoder(AudioEncoder):
    """Shared plumbing for the Qwen audio towers (AuT) used by Qwen3-ASR and Qwen-Omni.

    Both towers consume the *flat varlen* form -- the valid mel frames of every
    utterance concatenated along time, plus ``feature_lens`` -- and return a flat
    ``(sum_frames, output_dim)`` tensor. Subclasses split that back per utterance;
    :meth:`_pad_chunks` turns the pieces into the padded ``(B, T, H)`` + frame mask
    that :class:`~glide.models.speech_llm.SpeechLLM` expects.
    """

    input_kind = "input_features"

    def __init__(self, tower, feature_extractor, output_dim: int, freeze: bool):
        super().__init__()
        self.model = tower
        self.feature_extractor = feature_extractor
        self.output_dim = output_dim
        if freeze:
            _freeze_module(self.model)

    @staticmethod
    def _compute_feature_lengths(input_features, feature_attention_mask):
        """Return the number of valid mel frames per utterance, shape ``(B,)``.

        The mask may be at sample resolution (WhisperFeatureExtractor emits e.g. 480000
        entries) or frame resolution; the length ratio maps either to mel frames.
        """
        n_frames = input_features.shape[-1]
        if feature_attention_mask is not None:
            mask_len = feature_attention_mask.shape[1]
            valid = feature_attention_mask.sum(dim=1).float()
            return (valid * n_frames / mask_len).round().long().clamp(1, n_frames)
        return torch.full((input_features.shape[0],), n_frames,
                          device=input_features.device, dtype=torch.long)

    @staticmethod
    def _concat_valid_frames(input_features, feat_lens):
        """Concatenate each utterance's valid frames along time: ``(n_mels, sum_frames)``."""
        return torch.cat(
            [input_features[b][:, : int(feat_lens[b])] for b in range(input_features.shape[0])],
            dim=1,
        )

    @staticmethod
    def _get_hidden_states(output):
        return output.last_hidden_state if hasattr(output, "last_hidden_state") else output

    @staticmethod
    def _pad_chunks(chunks, device):
        """Right-pad per-utterance ``(L_i, H)`` chunks into ``(B, max L, H)`` + a frame mask."""
        lmax = max(c.shape[0] for c in chunks)
        hidden = chunks[0].new_zeros(len(chunks), lmax, chunks[0].shape[-1])
        mask = torch.zeros(len(chunks), lmax, dtype=torch.long, device=device)
        for b, c in enumerate(chunks):
            hidden[b, : c.shape[0]] = c
            mask[b, : c.shape[0]] = 1
        return hidden, mask


class Qwen3ASRAudioTowerEncoder(_QwenAudioTowerEncoder):
    """Qwen3-ASR's audio tower (``thinker.audio_tower``); built by ``qwen3_asr_aut``.

    Args:
        output_lengths_fn: Qwen3-ASR's ``_get_feat_extract_output_lengths``, mapping
            mel frames to post-CNN output frames (used to split the varlen output).
        use_varlen: Run all utterances in one FA2 varlen call. When ``False`` (no
            flash-attn) each utterance is encoded separately, which is also leak-free.
    """

    def __init__(self, tower, feature_extractor, output_dim: int, freeze: bool, *,
                 output_lengths_fn, use_varlen: bool):
        super().__init__(tower, feature_extractor, output_dim, freeze)
        self._output_lengths_fn = output_lengths_fn
        self.use_varlen = use_varlen

    def forward(self, input_features, feature_attention_mask=None, **_):
        feat_lens = self._compute_feature_lengths(input_features, feature_attention_mask)
        if self.use_varlen:
            # Varlen (FA2): ONE call over all utterances concatenated. The AuT builds
            # cu_seqlens from feature_lens and FA2 attends block-diagonally per utterance
            # (verified leak-free: audio0 output is unchanged when other audios change).
            feats = self._concat_valid_frames(input_features, feat_lens)
            flat = self._get_hidden_states(self.model(feats, feature_lens=feat_lens))
            chunks = list(flat.split(self._output_lengths_fn(feat_lens).tolist(), dim=0))
            return self._pad_chunks(chunks, flat.device)
        # Fallback: per-audio loop (leak-free under any backend, but sequential/slower).
        outs = []
        for b in range(input_features.shape[0]):
            fl = int(feat_lens[b])
            out = self.model(input_features[b][:, :fl], feature_lens=feat_lens[b:b + 1])
            outs.append(self._get_hidden_states(out))
        return self._pad_chunks(outs, outs[0].device)


class QwenOmniAudioTowerEncoder(_QwenAudioTowerEncoder):
    """Qwen2.5-/Qwen3-Omni's audio tower; built by ``qwen_omni_aut``.

    The Omni tower requires an explicit ``feature_lens`` (without it,
    ``chunk_and_pad_features(..., None, ...)`` crashes) and always runs varlen.
    """

    def forward(self, input_features, feature_attention_mask=None, **_):
        feat_lens = self._compute_feature_lengths(input_features, feature_attention_mask)
        feats = self._concat_valid_frames(input_features, feat_lens)
        flat = self._get_hidden_states(self.model(feats, feature_lens=feat_lens))
        _, out_lens = self.model._get_feat_extract_output_lengths(feat_lens)
        chunks = list(flat.split([int(x) for x in out_lens], dim=0))
        return self._pad_chunks(chunks, flat.device)


def _force_flash_attention(tower) -> bool:
    """Switch ``tower`` and its submodules to FA2 if flash-attn is installed.

    Returns whether the tower now runs FA2 (i.e. whether the varlen path is usable).
    """
    import importlib.util

    if importlib.util.find_spec("flash_attn") is None:
        return False
    for module in [tower, *tower.modules()]:
        sub_config = getattr(module, "config", None)
        if sub_config is not None:
            try:
                sub_config._attn_implementation = "flash_attention_2"
            except Exception:
                pass
        if hasattr(module, "_attn_implementation"):
            module._attn_implementation = "flash_attention_2"
    return getattr(tower.config, "_attn_implementation", None) == "flash_attention_2"


@audio_encoders.register("qwen3_asr_aut", exist_ok=True)
def build_qwen3_asr_audio_tower(cfg: AudioEncoderConfig, sample_rate: int = 16000):
    """Qwen3-ASR audio tower (requires the ``qwen_asr`` package).

    Extracts ``.thinker.audio_tower`` from a Qwen3-ASR checkpoint and its processor's
    feature extractor. Input is log-mel ``input_features`` + ``feature_attention_mask``.
    """
    import importlib

    from transformers import AutoConfig, AutoModel, AutoProcessor

    # Optional dependency: import dynamically so a static checker / a bare install
    # without the qwen_asr package doesn't choke on this builder. Register the
    # custom arch with the Auto classes so from_pretrained works (self-contained;
    # no separate plugin needed just for the encoder).
    backend = importlib.import_module("qwen_asr.core.transformers_backend")
    AutoConfig.register("qwen3_asr", backend.Qwen3ASRConfig, exist_ok=True)
    AutoModel.register(backend.Qwen3ASRConfig, backend.Qwen3ASRForConditionalGeneration,
                       exist_ok=True)

    repo = cfg.pretrained or "Qwen/Qwen3-ASR-1.7B"
    full = AutoModel.from_pretrained(repo, **cfg.extra_kwargs)
    tower = full.thinker.audio_tower
    out_dim = int(getattr(tower.config, "output_dim", None)
                  or full.config.get_text_config().hidden_size)

    # The AuT only honors cu_seqlens (true per-utterance block-diagonal attention) under
    # flash_attention_2; under sdpa/eager a concatenated call leaks across utterances. So
    # force FA2 on the tower -> enables the fast leak-free *varlen* path (one call for the
    # whole batch). If FA2 isn't available, fall back to the per-audio loop (leak-free too).
    use_varlen = _force_flash_attention(tower)
    _log.info(
        "qwen3_asr_aut: %s",
        "FA2 varlen (batched, leak-free)" if use_varlen else "per-audio loop (sdpa fallback)",
    )
    return Qwen3ASRAudioTowerEncoder(
        tower,
        AutoProcessor.from_pretrained(repo).feature_extractor,
        out_dim,
        cfg.freeze,
        output_lengths_fn=backend.modeling_qwen3_asr._get_feat_extract_output_lengths,
        use_varlen=use_varlen,
    )


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
def build_qwen_omni_audio_tower(cfg: AudioEncoderConfig, sample_rate: int = 16000):
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
    return QwenOmniAudioTowerEncoder(
        tower, AutoFeatureExtractor.from_pretrained(repo), out_dim, cfg.freeze
    )
