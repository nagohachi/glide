"""Base model, PEFT/LoRA and special-token configuration."""

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["ModelConfig", "PeftConfigSpec", "SpecialTokensConfig"]


@dataclass
class ModelConfig:
    """How to load the base model, tokenizer and (multimodal) processor."""

    #: Required. HF hub id or local path; ``""`` means "not set".
    model_name_or_id: str = ""
    #: Defaults to ``model_name_or_id`` when unset.
    tokenizer_name_or_id: str | None = None
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
