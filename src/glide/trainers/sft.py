"""Supervised fine-tuning (SFT) trainer construction.

Text SFT uses TRL's ``SFTTrainer`` directly on a conversational prompt/completion
dataset (the trainer applies the chat template and completion-only masking).
Speech/vision SFT uses :class:`~glide.trainers.length_sampler_trainer.LengthGroupedSFTTrainer`
with a :class:`~glide.data.collator.MultimodalSFTCollator` and TRL's
``skip_prepare_dataset`` so our collator owns tokenization + media features.
"""

from trl import SFTConfig, SFTTrainer

from ..config.schema import GlideConfig, Modality
from ..config.loader import build_training_args
from ..data.build import (
    build_multimodal_dataset,
    build_sft_text_dataset,
    compute_audio_lengths,
)
from ..data.collator import MultimodalSFTCollator
from ..data.jsonl import read_jsonl
from ..models.loader import load_model_and_processor
from .common import init_plugins, maybe_generation_callback, maybe_test_callback
from .length_sampler_trainer import LengthGroupedSFTTrainer

__all__ = ["build_sft_trainer"]


def _build_collator(config: GlideConfig, loaded):
    """Build the multimodal collator: composed-speech, plugin, or the default."""
    if config.modality is Modality.SPEECH and config.speech.encoder.name:
        from ..data.composed_collator import ComposedSpeechCollator

        audio_token = config.special_tokens.audio_token
        if not audio_token:
            raise ValueError("composed Speech-LLM requires special_tokens.audio_token")
        encoder = loaded.model.encoder
        return ComposedSpeechCollator(
            tokenizer=loaded.tokenizer,
            feature_extractor=encoder.feature_extractor,
            input_kind=encoder.input_kind,
            audio_token=audio_token,
            data=config.data,
            template=config.template,
            sample_rate=config.speech.sample_rate,
            speed_perturb=config.speech.speed_perturb,
            specaugment=config.speech.specaugment,
            completion_only=config.template.train_on_completions_only,
        )
    if config.template.collator:
        from ..registry import collators

        return collators.get(config.template.collator)(config, loaded.processor)
    return MultimodalSFTCollator(
        processor=loaded.processor,
        data=config.data,
        template=config.template,
        modality=config.modality,
        sample_rate=config.speech.sample_rate,
        completion_only=config.template.train_on_completions_only,
    )


def _peft_config(config: GlideConfig):
    if not config.peft.enabled:
        return None
    from peft import LoraConfig

    p = config.peft
    target_modules = p.target_modules
    modules_to_save = list(p.modules_to_save or [])
    if config.modality is Modality.SPEECH and config.speech.encoder.name:
        # Composed SpeechLLM is a plain nn.Module (not a PreTrainedModel). 'all-linear'
        # would LoRA-wrap EVERY nn.Linear in the tree: the from-scratch projector (its
        # random base weights frozen -> the modality bridge can never learn), the frozen
        # encoder's linears (unintended trainable adapters), and lm_head. Instead target
        # only the LLM's attention/MLP projections via a regex anchored on `llm.` (so the
        # encoder's identically-named q_proj/... are excluded), and keep the projector
        # fully trainable through modules_to_save.
        if target_modules == "all-linear" or target_modules is None:
            target_modules = (
                r"llm\.model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
            )
        if "projector" not in modules_to_save:
            modules_to_save.append("projector")
    return LoraConfig(
        r=p.r,
        lora_alpha=p.lora_alpha,
        lora_dropout=p.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save or None,
        bias=p.bias,
        task_type="CAUSAL_LM",
    )


def _augment_training_defaults(config: GlideConfig) -> None:
    """Populate SFT-specific ``training`` defaults derived from structured config."""
    t = config.training
    t.setdefault("max_length", config.template.max_length)
    t.setdefault("packing", config.packing.enabled)
    if config.packing.enabled:
        t.setdefault("packing_strategy", config.packing.strategy)

    if config.modality is Modality.TEXT:
        t.setdefault("completion_only_loss", config.template.train_on_completions_only)
    else:
        # Multimodal: our collator owns preprocessing.
        dk = dict(t.get("dataset_kwargs", {}))
        dk.setdefault("skip_prepare_dataset", True)
        t["dataset_kwargs"] = dk
        t.setdefault("remove_unused_columns", False)
        # Composed SpeechLLM is a plain nn.Module whose LLM ties embeddings<->lm_head
        # (shared storage); safetensors refuses shared tensors, so use torch.save.
        if config.modality is Modality.SPEECH and config.speech.encoder.name:
            t.setdefault("save_safetensors", False)


def build_sft_trainer(config: GlideConfig):
    """Build a ready-to-train SFT trainer from ``config``."""
    init_plugins(config)
    loaded = load_model_and_processor(config)

    _augment_training_defaults(config)
    args: SFTConfig = build_training_args(config, SFTConfig)

    callbacks = []
    eval_records = None
    if config.eval.generate.enabled and config.data.eval is not None:
        paths = config.data.eval if isinstance(config.data.eval, list) else [config.data.eval]
        eval_records = []
        for p in paths:
            eval_records.extend(read_jsonl(p))
        # Cap AR-eval records (generation is slow); matches the teacher-forced eval cap.
        if config.data.max_eval_samples is not None:
            eval_records = eval_records[: config.data.max_eval_samples]
    gen_cb = maybe_generation_callback(config, loaded.processor, eval_records)
    if gen_cb is not None:
        callbacks.append(gen_cb)
    test_cb = maybe_test_callback(config, loaded.processor)
    if test_cb is not None:
        callbacks.append(test_cb)

    if config.modality is Modality.TEXT:
        trainer = SFTTrainer(
            model=loaded.model,
            args=args,
            train_dataset=build_sft_text_dataset(config, "train"),
            eval_dataset=build_sft_text_dataset(config, "eval"),
            processing_class=loaded.tokenizer,
            peft_config=_peft_config(config),
            callbacks=callbacks or None,
        )
    else:
        train_ds = build_multimodal_dataset(config, "train")
        eval_ds = build_multimodal_dataset(config, "eval")
        collator = _build_collator(config, loaded)
        lengths = None
        if config.modality is Modality.SPEECH and config.speech.length_grouped_sampler:
            lengths = compute_audio_lengths(train_ds, config)
        trainer = LengthGroupedSFTTrainer(
            model=loaded.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            processing_class=loaded.processor,
            peft_config=_peft_config(config),
            callbacks=callbacks or None,
            glide_lengths=lengths,
            glide_batch_size=args.per_device_train_batch_size,
            glide_max_tokens=config.speech.max_tokens_per_batch,
            glide_seed=config.seed,
            glide_warmup=config.speech.warmup,
            glide_target_lr=config.training.get("learning_rate", args.learning_rate),
            glide_warmup_ratio=config.training.get("warmup_ratio", 0.0),
            glide_composed=bool(config.speech.encoder.name),
        )

    # Let the generation callback log through the trainer.
    for cb in callbacks:
        cb._trainer = trainer
    return trainer
