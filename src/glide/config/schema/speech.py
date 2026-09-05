"""Speech modality: encoder, projector, augmentation and warmup."""

from dataclasses import dataclass, field
from typing import Any
from .enums import SpeechTask

__all__ = ["AudioEncoderConfig", "ProjectorConfig", "SpecAugmentConfig", "SpeedPerturbConfig", "AudioAugmentConfig", "ProjectorWarmupConfig", "SpeechConfig"]


@dataclass
class AudioEncoderConfig:
    """Audio encoder selection for a composed Speech-LLM.

    ``name`` resolves a builder in :data:`glide.registry.audio_encoders`. Built-ins:

    * ``whisper`` -- OpenAI Whisper encoder (``pretrained`` = ``openai/whisper-{small,
      medium,large,large-v2,large-v3}`` or any Whisper id). Log-mel input.
    * ``wavlm`` -- ``microsoft/wavlm-{base,large}``. Raw-waveform input.
    * ``xls_r`` -- ``facebook/wav2vec2-xls-r-{300m,1b,2b}``. Raw-waveform input.
    * ``qwen3_asr_aut`` -- Qwen3-ASR audio tower (requires the ``qwen_asr`` package).
    * ``qwen_omni_aut`` -- Qwen2.5/Qwen3-Omni audio tower (transformers).

    ``None`` ``name`` means "use the base model's built-in audio tower" (the plain
    :func:`load_model_and_processor` path, e.g. a stock Qwen3-ASR checkpoint).
    """

    name: str | None = None
    #: HF hub id / local path / size of the pretrained encoder.
    pretrained: str | None = None
    #: Freeze encoder parameters (common for projector warm-up).
    freeze: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectorConfig:
    """Projector bridging encoder hidden size -> LLM embedding size."""

    #: ``mlp_gelu`` (from scratch) | ``qwen3_asr_proj`` | ``qwen_omni_proj``.
    name: str = "mlp_gelu"
    #: For pretrained projectors: where to load weights from (HF id / path).
    pretrained: str | None = None
    #: MLP hidden size and depth (``mlp_gelu``).
    hidden_dim: int | None = None
    num_layers: int = 2
    #: Downsample factor along time (stack/avg frames before projecting).
    downsample: int = 1
    freeze: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpecAugmentConfig:
    """SpecAugment on log-mel features (frequency + time masking)."""

    enabled: bool = False
    freq_mask_width: int = 40
    num_freq_mask: int = 2
    time_mask_width_ratio: float = 0.12
    num_time_mask: int = 5


@dataclass
class SpeedPerturbConfig:
    """Speed perturbation of the waveform (data augmentation).

    With ``from_field=True`` the per-record ``speed`` field (CSJ ``native_sft``
    style: ``{"audio": ..., "speed": 1.0, "text": ...}``) sets the factor; otherwise
    a factor is sampled uniformly from ``factors`` each epoch.
    """

    enabled: bool = False
    factors: list[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])
    from_field: bool = False
    field_name: str = "speed"


@dataclass
class AudioAugmentConfig:
    """Waveform / feature augmentations, applied to training batches only."""

    #: Feature-domain masking on log-mel inputs.
    specaugment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)
    #: Waveform-domain resampling.
    speed_perturb: SpeedPerturbConfig = field(default_factory=SpeedPerturbConfig)


@dataclass
class ProjectorWarmupConfig:
    """Two-phase ``projector-only warmup -> full fine-tuning`` schedule.

    DDP-safe: all params stay trainable (no ``requires_grad`` toggling); the
    optimizer has two groups (projector / rest) and a per-group LR schedule holds
    the ``rest`` group at LR 0 during phase 1, so only the projector updates.

    * Phase 1 (step < ``projector_only_steps``): projector LR ramps 0 ->
      ``projector_lr``; everything else stays at LR 0.
    * Phase 2 (step >= ``projector_only_steps``): all params ramp 0 ->
      ``training.learning_rate`` (over ``full_warmup_steps`` or
      ``training.warmup_ratio * (T - P)``), then linear decay to 0.
    """

    #: Length of the projector-only phase, in steps. ``0`` disables the schedule.
    projector_only_steps: int = 0
    #: Peak LR for the projector during phase 1 (also the optimizer base LR).
    #: Required when ``projector_only_steps > 0``. ``-1`` means "not set".
    projector_lr: float = -1.0
    #: Phase-2 LR-warmup length in steps; ``0`` -> ``warmup_ratio * (T - P)``.
    full_warmup_steps: int = 0
    #: Parameter-name substrings identifying the projector. Defaults cover the
    #: Qwen3-ASR audio tower (``audio_tower.proj1/proj2``) and the composed
    #: Speech-LLM (``projector.``).
    projector_patterns: list[str] = field(
        default_factory=lambda: ["audio_tower.proj1.", "audio_tower.proj2.", "projector."]
    )


@dataclass
class SpeechConfig:
    """Speech-modality settings, including composable encoder/projector/LLM.

    A *composed* Speech-LLM = ``encoder`` (audio -> hidden) + ``projector`` (hidden
    -> LLM embedding) + the base text LLM in :class:`ModelConfig` (Qwen3-1.7B,
    Llama-3.2-1B/3B, Gemma-3-1B/4B, ...). Leave ``encoder.name`` unset to instead
    use a model that already bundles its own audio tower (e.g. stock Qwen3-ASR).
    """

    enabled: bool = False
    task: SpeechTask = SpeechTask.RECOGNITION
    #: Target sample rate; audio is resampled on load.
    sample_rate: int = 16000
    #: Composed-model components (see their dataclasses).
    encoder: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    #: Length-based batch sampler: group similar-length samples to reduce padding.
    #: The order *within* the epoch is reshuffled every epoch (see
    #: :class:`glide.data.sampler.LengthGroupedBatchSampler`).
    length_grouped_sampler: bool = True
    #: Bucket granularity in frames/samples for the length sampler.
    length_bucket_size: int = 0
    #: Dynamic batching budget: max summed length (audio samples) per batch. When
    #: set, the per-device batch size *varies* (more short utterances per batch).
    #: ``per_device_train_batch_size`` then acts as a hard cap on the count.
    max_tokens_per_batch: int | None = None
    #: Training-time audio augmentations (see :class:`AudioAugmentConfig`).
    augment: AudioAugmentConfig = field(default_factory=AudioAugmentConfig)
    #: Two-phase projector-warmup -> full fine-tuning schedule.
    warmup: ProjectorWarmupConfig = field(default_factory=ProjectorWarmupConfig)
