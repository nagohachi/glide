"""Typed configuration schema for :mod:`glide`.

The configuration is a tree of small, composable dataclasses. Every field maps
to a YAML key, and any field can be overridden on the command line with a dotted
path (e.g. ``--model.name Qwen/Qwen3-1.7B`` or ``--training.learning_rate 1e-5``).

The ``training`` section is intentionally a *free-form* dictionary rather than a
dataclass: it is forwarded to the relevant TRL config object
(:class:`trl.SFTConfig`, :class:`trl.GRPOConfig`) so that every TRL/transformers
``TrainingArguments`` field is accepted without having to mirror hundreds of fields
here. See :func:`glide.config.loader.build_training_args`.
"""

import dataclasses
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, Literal


class Task(StrEnum):
    """Supported post-training tasks (selected by the CLI subcommand)."""

    SFT = "sft"
    GRPO = "grpo"
    GSPO = "gspo"


class Modality(StrEnum):
    """Input modality of the model being trained."""

    TEXT = "text"
    SPEECH = "speech"
    VISION = "vision"


class SpeechTask(StrEnum):
    """Sub-task for the speech modality (controls validation metrics)."""

    RECOGNITION = "recognition"
    TRANSLATION = "translation"


@dataclass
class ModelConfig:
    """How to load the base model, tokenizer and (multimodal) processor."""

    #: Required. HF hub id or local path; ``""`` means "not set".
    name: str = ""
    #: Defaults to the model path when unset.
    tokenizer_name: str | None = None
    #: Required. ``flash_attention_2`` | ``sdpa`` | ``eager``. ``flash_attention_2``
    #: requires the ``flash-attn`` extra and a compatible GPU. ``""`` means "not set".
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager", ""] = ""
    #: ``bfloat16`` | ``float16`` | ``float32`` | ``auto``.
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = False
    #: Load in 4/8-bit (requires bitsandbytes). ``None`` disables quantization.
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    #: Also settable under ``training``.
    gradient_checkpointing: bool = False
    #: Optional explicit AutoModel class name to use (e.g.
    #: ``AutoModelForImageTextToText``). When ``None`` it is inferred from the
    #: modality. See :mod:`glide.models.loader`.
    auto_class: str | None = None
    #: Extra kwargs forwarded verbatim to ``from_pretrained``.
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    #: Optional path to a saved *state dict* (``pytorch_model.bin``) loaded over the
    #: assembled model with ``strict=False`` after construction. Use it to initialise
    #: a composed Speech-LLM from a prior glide checkpoint (e.g. an SFT checkpoint as
    #: the starting policy for GSPO) without it being a self-contained HF model dir.
    state_dict_path: str | None = None


@dataclass
class PeftConfigSpec:
    """LoRA / PEFT settings. ``enabled=False`` performs full fine-tuning."""

    enabled: bool = False
    #: Required when ``enabled``. ``-1`` means "not set".
    r: int = -1
    lora_alpha: int = -1
    #: PEFT's own default. The LoRA paper reports 0.1 for its GLUE runs; 0.05 is
    #: the QLoRA convention, not an original-paper value.
    lora_dropout: float = 0.0
    #: Required when ``enabled``. ``None`` means "not set".
    target_modules: list[str] | str | None = None
    modules_to_save: list[str] = field(default_factory=list)
    bias: Literal["none", "all", "lora_only"] = "none"


@dataclass
class SpecialTokensConfig:
    """Special-token handling, configured entirely from YAML.

    Example::

        special_tokens:
          additional: ["<audio>", "<audio_pad>", "<image>", "<image_pad>"]
          audio_token: "<audio>"
          audio_pad_token: "<audio_pad>"
          image_token: "<image>"
          image_pad_token: "<image_pad>"
          resize_embeddings: true
          pad_to_multiple_of: 8
    """

    #: Additional special tokens to add to the tokenizer vocabulary.
    additional: list[str] = field(default_factory=list)
    #: Override standard special tokens if needed.
    bos_token: str | None = None
    eos_token: str | None = None
    pad_token: str | None = None
    #: Semantic multimodal placeholders. ``*_token`` is the single marker that
    #: appears in text; ``*_pad_token`` is the token whose embeddings are replaced
    #: by encoder features at the placeholder positions.
    audio_token: str | None = None
    audio_pad_token: str | None = None
    image_token: str | None = None
    image_pad_token: str | None = None
    #: Resize the input/output embedding matrices after adding tokens.
    resize_embeddings: bool = True
    #: Pad the resized vocab to a multiple of this (kernel-friendly).
    pad_to_multiple_of: int | None = 8


@dataclass
class DataConfig:
    """JSONL dataset locations and field mapping."""

    #: Corpus name selecting an absolute root from the top-level ``data_roots`` map
    #: (machine-specific; keep it in a gitignored ``data_root.yaml`` that runs
    #: ``extends``, see ``configs/data_root.example.yaml``). Relative ``train``/``eval``
    #: paths are resolved under that root.
    corpus: str | None = None
    #: Explicit absolute data root (alternative to ``corpus``); prepended to relative
    #: ``train``/``eval`` paths. ``corpus`` takes precedence when both are set.
    root: str | None = None
    #: Path(s) to training JSONL file(s) (relative to the corpus root unless absolute).
    train: str | list[str] | None = None
    #: Path(s) to evaluation/validation JSONL file(s) (relative to the corpus root).
    eval: str | list[str] | None = None
    #: Path(s) to held-out test JSONL file(s) evaluated once at the end of training.
    test: str | list[str] | None = None
    #: Read records lazily by byte offset instead of loading into memory.
    lazy: bool = True
    #: Field names in each JSONL record.
    messages_field: str = "messages"  # chat-style records
    prompt_field: str = "prompt"  # prompt/response or RL records
    response_field: str = "response"
    #: For speech/vision: the field holding the media path / array.
    audio_field: str = "audio"
    image_field: str = "image"
    #: Optional per-record fields carrying the audio length for the length-grouped
    #: sampler, avoiding a filesystem header read per utterance. ``duration_field`` is
    #: in seconds (multiplied by ``speech.sample_rate``); ``num_samples_field`` is the
    #: raw sample count. Either, when present on a record, is used verbatim.
    duration_field: str | None = None
    num_samples_field: str | None = None
    #: Optional reference text field used by validation metrics (WER/CER/BLEU).
    reference_field: str = "reference"
    #: Field holding the assistant reasoning/CoT for thinking-mode SFT. When present
    #: (and ``template.enable_thinking`` is True) it is passed as the chat-template
    #: ``reasoning_content`` so the target becomes ``<think>{reasoning}</think>{answer}``.
    reasoning_field: str = "reasoning"
    #: Cap the number of samples (debugging). ``None`` = all.
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    #: Dataloader worker processes. ``0`` loads in the main process (the PyTorch
    #: default); >0 forks workers, which the lazy JSONL reader is fork-safe for.
    num_workers: int = 0


@dataclass
class TemplateConfig:
    """Prompt/chat-template construction and loss masking."""

    #: Named template in the registry, or ``None`` to use the tokenizer's
    #: built-in chat template.
    name: str | None = None
    #: Train only on the assistant/response tokens (completion-only masking).
    train_on_completions_only: bool = True
    #: When chat templates are used, the substring that marks the start of the
    #: assistant turn (used to locate the response span for masking).
    response_template: str | None = None
    instruction_template: str | None = None
    #: Add a generation prompt suffix when formatting prompts for generation.
    add_generation_prompt: bool = True
    #: System prompt prepended to every conversation when one is not present.
    system_prompt: str | None = None
    max_length: int = 2048
    #: Thinking mode for models with a reasoning toggle (e.g. Qwen3). ``True`` keeps
    #: ``<think>`` reasoning (SFT supervises ``<think>...</think>`` + answer);
    #: ``False`` injects an empty ``<think></think>`` (SFT supervises the answer only);
    #: ``None`` uses the template default. Passed to ``apply_chat_template`` for both
    #: the prompt prefix and the target so completion masking stays aligned.
    enable_thinking: bool | None = None
    #: Name of a registered custom data collator (plugin) to use for the
    #: multimodal SFT path instead of the built-in :class:`MultimodalSFTCollator`.
    #: The builder is called as ``builder(config, processor)``.
    collator: str | None = None


@dataclass
class PackingConfig:
    """Sequence packing settings (forwarded to TRL when applicable)."""

    enabled: bool = False
    #: ``ffd`` (first-fit-decreasing) or ``wrapped``; matches TRL's options.
    strategy: Literal["ffd", "wrapped"] = "ffd"
    max_length: int | None = None


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
    #: SpecAugment and speed-perturbation augmentations.
    specaugment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)
    speed_perturb: SpeedPerturbConfig = field(default_factory=SpeedPerturbConfig)
    #: Two-phase projector-warmup -> full fine-tuning schedule.
    warmup: ProjectorWarmupConfig = field(default_factory=ProjectorWarmupConfig)


@dataclass
class VisionConfig:
    """Vision-modality specific settings (vision + text input).

    The vision path is model-agnostic: it relies on the HF ``AutoProcessor`` and
    ``AutoModelForImageTextToText`` so any image-text-to-text model plugs in.
    """

    enabled: bool = False
    #: Longest edge / max pixels passed to the image processor (model dependent).
    max_pixels: int | None = None
    min_pixels: int | None = None


@dataclass
class RewardSpec:
    """A single reward function entry for RL training.

    ``name`` resolves a function in the reward registry (built-in or plugin).
    ``weight`` scales its contribution; ``kwargs`` are passed at construction.
    """

    name: str
    weight: float = 1.0
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RLConfig:
    """Reinforcement-learning settings (GRPO / GSPO)."""

    #: Reward functions (composed as a weighted sum). Required for GRPO/GSPO.
    rewards: list[RewardSpec] = field(default_factory=list)
    #: Use a vLLM rollout server for generation (``server`` / ``colocate``).
    use_vllm: bool = False
    vllm_mode: str = "server"
    #: Prefill string prepended to every generation (e.g. a CoT opener).
    response_prefix: str | None = None

    # --- Speech GSPO loop (glide.trainers.rl_speech) ---------------------- #
    # These drive the self-contained speech-in-the-loop GSPO/GRPO trainer, which
    # generates with audio (text-only TRL GRPO can't) and rolls out through a vLLM
    # server on a *separate* GPU from training. Ignored by the text TRL path.
    #: Number of sampled completions per prompt (the GRPO/GSPO group size G).
    num_generations: int = 8
    #: Rollout sampling temperature / nucleus cutoff.
    temperature: float = 1.0
    top_p: float = 1.0
    #: Max new tokens per rollout completion.
    max_completion_length: int = 256
    #: Clip range eps for the (sequence-level, GSPO) policy ratio.
    clip_eps: float = 0.2
    #: KL-to-reference penalty coefficient (0 disables the reference model entirely).
    kl_beta: float = 0.0
    #: Group-normalise advantages by their std (GRPO ``scale_rewards``).
    scale_rewards: bool = True
    #: Inner policy updates per rollout batch (mu). 1 == on-policy (ratio == 1).
    num_iterations: int = 1
    #: vLLM rollout server address (``server`` mode). The server runs the *same*
    #: composed model on a different GPU; weights are pushed after each update.
    vllm_server_host: str = "localhost"
    vllm_server_port: int = 8000
    #: Refresh the vLLM server weights every N optimizer steps (1 == every step).
    vllm_sync_every: int = 1
    #: Shared path where the trainer writes the remapped policy weights for the vLLM
    #: server to reload (weight sync). Must be readable by the server host.
    vllm_sync_path: str = "outputs/vllm_sync_policy.safetensors"
    #: Reward field on each record consumed by the CER reward (the reference text).
    reference_key: str = "reference"


@dataclass
class GenerationConfig:
    """Decoding parameters for validation-time autoregressive decoding."""

    enabled: bool = False
    max_new_tokens: int = 256
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    #: Batch size for generation during evaluation.
    batch_size: int = 8


@dataclass
class EvalConfig:
    """Validation behaviour, including AR decoding and metrics."""

    #: Run autoregressive decoding at validation time (speech/vision/LLM).
    generate: GenerationConfig = field(default_factory=GenerationConfig)
    #: Metrics to compute on decoded text: any of
    #: ``wer`` ``cer`` ``bleu`` ``rouge`` (or registered custom metrics).
    metrics: list[str] = field(default_factory=list)
    #: Normalize text before scoring (lowercase, strip punctuation).
    normalize_text: bool = True
    #: Score only the text *after* the last occurrence of this literal delimiter
    #: (e.g. ``"</think>"`` to drop a thinking block before computing WER/CER).
    #: If the delimiter is absent from a string, that string is scored whole.
    #: Applied to both hypotheses and references. ``None`` disables it.
    answer_after: str | None = None
    #: Regex whose **group 1** (or whole match if it has no groups) selects the
    #: substring to score. Applied after ``answer_after`` if both are set.
    #: If the pattern does not match a string, that string is scored whole.
    #: Applied to both hypotheses and references. ``None`` disables it.
    answer_regex: str | None = None


@dataclass
class DistributedConfig:
    """Multi-GPU / multi-node launch settings (driven from YAML, not env vars).

    ``glide <task>`` self-launches under ``torch.distributed.run`` when
    ``nproc_per_node`` resolves to > 1. ``nproc_per_node: null`` (default) means
    *auto* -- use all visible GPUs (``torch.cuda.device_count()``).
    """

    #: GPUs (processes) per node. ``None`` = auto-detect = number of visible GPUs.
    nproc_per_node: int | None = None
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str | None = None
    master_port: int | None = None


@dataclass
class LoggingConfig:
    """Experiment logging (wandb / tensorboard)."""

    #: Subset of ``["wandb", "tensorboard"]`` (or ``["none"]``).
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    project: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class GlideConfig:
    """Root configuration object assembled from the merged YAML + CLI overrides."""

    task: Task = Task.SFT
    modality: Modality = Modality.TEXT
    seed: int = 42

    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PeftConfigSpec = field(default_factory=PeftConfigSpec)
    special_tokens: SpecialTokensConfig = field(default_factory=SpecialTokensConfig)
    data: DataConfig = field(default_factory=DataConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    #: Machine-specific corpus name -> absolute data root map, selected by
    #: ``data.corpus``. Keep in a gitignored ``data_root.yaml`` pulled in via
    #: ``extends`` (see ``configs/data_root.example.yaml``).
    data_roots: dict[str, str] = field(default_factory=dict)

    #: Module paths (dotted) or file paths to import for plugin registration,
    #: e.g. ``["src.my_rewards", "src/my_encoder.py"]``.
    plugins: list[str] = field(default_factory=list)

    #: Free-form arguments forwarded to the TRL config object. ``output_dir`` here
    #: is versioned to ``output_dir/v{N}-{datetime}`` at run start.
    training: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain (YAML-serializable) dict of the whole config."""
        return _to_plain(dataclasses.asdict(self))


def _to_plain(obj: Any) -> Any:
    """Recursively convert enums to their values for serialization."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj
