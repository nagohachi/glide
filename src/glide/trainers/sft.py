"""Supervised fine-tuning (SFT) trainer construction.

Text SFT uses TRL's ``SFTTrainer`` directly on a conversational prompt/completion
dataset (the trainer applies the chat template and completion-only masking).
Speech/vision SFT uses :class:`~glide.trainers.length_sampler_trainer.LengthGroupedSFTTrainer`
with a :class:`~glide.data.collator.MultimodalSFTCollator` and TRL's
``skip_prepare_dataset`` so our collator owns tokenization + media features.
"""

import dataclasses

from trl import SFTConfig, SFTTrainer

from ..config.loader import build_training_args
from ..config.schema import GlideConfig, Modality
from ..data.build import (
    build_multimodal_dataset,
    build_sft_text_dataset,
    compute_audio_lengths,
)
from ..data.collator import MultimodalSFTCollator
from ..models.loader import load_model_and_processor
from .common import build_eval_callbacks, build_peft_config, init_plugins
from .length_sampler_trainer import LengthGroupedSFTTrainer

__all__ = ["build_sft_trainer"]


def _build_collator(config: GlideConfig, loaded):
    """Build the multimodal collator: composed-speech, plugin, or the default."""
    if config.is_composed_speech_llm():
        from ..data.composed_collator import ComposedSpeechCollator

        return ComposedSpeechCollator.from_config(
            config, loaded.tokenizer, loaded.model.encoder, train=True
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


def _apply_sft_training_defaults(config: GlideConfig) -> None:
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
        # `save_safetensors` was removed from TrainingArguments in transformers 5.12;
        # only set it when the field still exists (else build_training_args rejects it).
        if config.is_composed_speech_llm():
            if any(f.name == "save_safetensors" for f in dataclasses.fields(SFTConfig)):
                t.setdefault("save_safetensors", False)


def build_sft_trainer(config: GlideConfig):
    """Build a ready-to-train SFT trainer from ``config``."""
    init_plugins(config)
    loaded = load_model_and_processor(config)

    _apply_sft_training_defaults(config)
    args: SFTConfig = build_training_args(config, SFTConfig)
    callbacks = build_eval_callbacks(config, loaded.processor)

    if config.modality is Modality.TEXT:
        trainer = SFTTrainer(
            model=loaded.model,
            args=args,
            train_dataset=build_sft_text_dataset(config, "train"),
            eval_dataset=build_sft_text_dataset(config, "eval"),
            processing_class=loaded.tokenizer,
            peft_config=build_peft_config(config),
            callbacks=callbacks or None,
        )
    else:
        train_ds = build_multimodal_dataset(config, "train")
        lengths = None
        if config.modality is Modality.SPEECH and config.speech.length_grouped_sampler:
            lengths = compute_audio_lengths(train_ds, config)
        trainer = LengthGroupedSFTTrainer(
            model=loaded.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=build_multimodal_dataset(config, "eval"),
            data_collator=_build_collator(config, loaded),
            processing_class=loaded.processor,
            peft_config=build_peft_config(config),
            callbacks=callbacks or None,
            glide_config=config,
            glide_lengths=lengths,
        )

    for callback in callbacks:
        callback.bind_trainer(trainer)
    return trainer
