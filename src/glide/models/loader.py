"""Model + processor loading: dtype, attention backend, special tokens, plugins.

The loader is modality-aware but model-agnostic:

* **text** / **speech** -> ``AutoModelForCausalLM`` by default (most speech-LLMs
  are decoder LMs with an audio adapter; override via ``model.auto_class``).
* **vision** -> ``AutoModelForImageTextToText`` with ``AutoProcessor`` (any HF
  image-text-to-text model plugs in).

Custom audio encoders / projectors registered as plugins are attached after load
when named in the config.
"""

from dataclasses import dataclass
from typing import Any

from ..config.schema import GlideConfig, Modality
from .special_tokens import SpecialTokenInfo, apply_special_tokens

__all__ = ["load_model_and_processor", "LoadedModel", "resolve_dtype"]

_DEFAULT_AUTO_CLASS = {
    Modality.TEXT: "AutoModelForCausalLM",
    Modality.SPEECH: "AutoModelForCausalLM",
    Modality.VISION: "AutoModelForImageTextToText",
}


@dataclass
class LoadedModel:
    """Bundle returned by :func:`load_model_and_processor`."""

    model: Any
    processor: Any  # processor (multimodal) or tokenizer (text)
    tokenizer: Any
    special_tokens: SpecialTokenInfo


def resolve_dtype(name: str):
    """Map a dtype name (``bfloat16`` etc.) to a ``torch.dtype`` or ``"auto"``."""
    import torch

    if name in ("auto", None):
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name]


def _quantization_config(model_cfg):
    if not (model_cfg.load_in_4bit or model_cfg.load_in_8bit):
        return None
    from transformers import BitsAndBytesConfig

    if model_cfg.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=resolve_dtype(model_cfg.torch_dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def _load_processor(model_cfg, modality: Modality, vision_cfg=None):
    """Load an ``AutoProcessor`` for multimodal models, else an ``AutoTokenizer``."""
    import transformers

    name = model_cfg.tokenizer_name or model_cfg.name
    kwargs = {"trust_remote_code": model_cfg.trust_remote_code}
    if modality in (Modality.SPEECH, Modality.VISION):
        proc_kwargs = dict(kwargs)
        # Forward the image-processor pixel budget for the vision path (the schema
        # advertises these as "passed to the image processor").
        if modality is Modality.VISION and vision_cfg is not None:
            if vision_cfg.max_pixels is not None:
                proc_kwargs["max_pixels"] = vision_cfg.max_pixels
            if vision_cfg.min_pixels is not None:
                proc_kwargs["min_pixels"] = vision_cfg.min_pixels
        try:
            return transformers.AutoProcessor.from_pretrained(name, **proc_kwargs)
        except Exception:
            pass  # fall through to tokenizer (e.g. text-only speech-LLM)
    return transformers.AutoTokenizer.from_pretrained(name, **kwargs)


def load_model_and_processor(config: GlideConfig) -> LoadedModel:
    """Load the base model and its processor/tokenizer per ``config``.

    For the speech modality with ``speech.encoder.name`` set, a *composed*
    Speech-LLM (encoder + projector + LLM) is assembled (see
    :func:`glide.models.speech_llm.build_speech_llm`). Otherwise the model is loaded
    directly (covers models with a built-in audio tower, e.g. stock Qwen3-ASR).
    """
    import transformers

    if config.modality is Modality.SPEECH and config.speech.encoder.name:
        from .speech_llm import build_speech_llm

        model, tok, info = build_speech_llm(config)
        _maybe_load_state_dict(model, config.model.state_dict_path)
        return LoadedModel(model=model, processor=tok, tokenizer=tok, special_tokens=info)

    model_cfg = config.model
    auto_class_name = model_cfg.auto_class or _DEFAULT_AUTO_CLASS[config.modality]
    auto_class = getattr(transformers, auto_class_name)

    load_kwargs: dict[str, Any] = {
        "dtype": resolve_dtype(model_cfg.torch_dtype),
        "attn_implementation": model_cfg.attn_implementation,
        "trust_remote_code": model_cfg.trust_remote_code,
        **model_cfg.extra_kwargs,
    }
    quant = _quantization_config(model_cfg)
    if quant is not None:
        load_kwargs["quantization_config"] = quant

    # Propagate attn_implementation to nested sub-configs. Composite models (e.g.
    # Qwen3-ASR: outer -> thinker -> text/audio configs) don't always inherit the
    # top-level setting, leaving the text decoder on sdpa even when
    # flash_attention_2 is requested -- which silently breaks FA2 sequence packing.
    hf_config = transformers.AutoConfig.from_pretrained(
        model_cfg.name, trust_remote_code=model_cfg.trust_remote_code
    )
    _propagate_attn_implementation(hf_config, model_cfg.attn_implementation)
    load_kwargs["config"] = hf_config

    model = auto_class.from_pretrained(model_cfg.name, **load_kwargs)
    processor = _load_processor(model_cfg, config.modality, config.vision)

    info = apply_special_tokens(processor, model, config.special_tokens)
    tokenizer = getattr(processor, "tokenizer", processor)

    _sanitize_generation_config(model)
    _maybe_load_state_dict(model, model_cfg.state_dict_path)

    return LoadedModel(model=model, processor=processor, tokenizer=tokenizer, special_tokens=info)


def _maybe_load_state_dict(model, path: str | None) -> None:
    """Load a saved state dict over ``model`` in place (``strict=False``), if given.

    Used to initialise a model from a prior glide checkpoint (``pytorch_model.bin``)
    -- e.g. an SFT checkpoint as the GSPO starting policy. ``strict=False`` tolerates
    benign key drift (the composed Speech-LLM saves encoder/projector/llm together).
    Missing/unexpected keys are reported so a silent mismatch can't pass unnoticed.
    """
    if not path:
        return
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and not any(
        k.startswith(("llm.", "encoder.", "projector.")) for k in state
    ):
        state = state["state_dict"]
    result = model.load_state_dict(state, strict=False)
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    print(f"[glide] loaded state_dict from {path} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print(f"[glide]   first missing: {missing[:8]}")
    if unexpected:
        print(f"[glide]   first unexpected: {unexpected[:8]}")


def _propagate_attn_implementation(hf_config, impl: str) -> None:
    """Recursively set ``_attn_implementation`` on a config and all nested configs.

    Needed for composite models whose sub-configs (text/audio towers) don't inherit
    the top-level ``attn_implementation`` — otherwise a requested ``flash_attention_2``
    silently stays ``sdpa`` on the inner decoder.
    """
    from transformers import PretrainedConfig

    if not isinstance(hf_config, PretrainedConfig):
        return
    hf_config._attn_implementation = impl
    if hasattr(hf_config, "_attn_implementation_internal"):
        hf_config._attn_implementation_internal = impl
    for value in list(vars(hf_config).values()):
        if isinstance(value, PretrainedConfig):
            _propagate_attn_implementation(value, impl)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _propagate_attn_implementation(item, impl)


def _sanitize_generation_config(model) -> None:
    """Reset sampling-only flags when ``do_sample`` is False so checkpoints save.

    Some hub models ship a ``generation_config.json`` with e.g. ``temperature``
    set while ``do_sample=False`` (Qwen3-ASR does). Recent transformers validate
    strictly on ``save_pretrained`` and refuse to save. Resetting the sampling
    flags to their defaults keeps greedy decoding unchanged and lets the model
    serialize.
    """
    gc = getattr(model, "generation_config", None)
    if gc is None or getattr(gc, "do_sample", False):
        return
    for attr, default in (("temperature", 1.0), ("top_p", 1.0), ("top_k", 50),
                          ("typical_p", 1.0), ("min_p", None)):
        if hasattr(gc, attr):
            setattr(gc, attr, default)


