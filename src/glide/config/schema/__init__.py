"""Typed configuration schema for :mod:`glide`.

The configuration is a tree of small, composable dataclasses. Every field maps
to a YAML key, and any field can be overridden on the command line with a dotted
path (e.g. ``--model.model_name_or_id Qwen/Qwen3-1.7B`` or
``--training.learning_rate 1e-5``).

The schema is split by area so each file stays readable on its own:

* :mod:`~glide.config.schema.enums` -- ``Task`` / ``Modality`` / ``SpeechTask``
* :mod:`~glide.config.schema.model` -- base model, PEFT, special tokens
* :mod:`~glide.config.schema.data` -- datasets, chat template, packing
* :mod:`~glide.config.schema.speech` -- encoder, projector, augmentation, warmup
* :mod:`~glide.config.schema.vision` -- vision-modality settings
* :mod:`~glide.config.schema.rl` -- GRPO / GSPO
* :mod:`~glide.config.schema.evaluation` -- decoding and metrics
* :mod:`~glide.config.schema.runtime` -- distributed launch, experiment logging
* :mod:`~glide.config.schema.root` -- :class:`GlideConfig`, which composes them

Every dataclass is re-exported here, so ``from glide.config.schema import
GlideConfig`` keeps working no matter which module a class lives in.

The ``training`` section is intentionally a *free-form* dictionary rather than a
dataclass: it is forwarded to the relevant TRL config object
(:class:`trl.SFTConfig`, :class:`trl.GRPOConfig`) so that every TRL/transformers
``TrainingArguments`` field is accepted without having to mirror hundreds of
fields here. See :func:`glide.config.loader.build_training_args`.
"""

from .data import DataConfig, PackingConfig, TemplateConfig
from .enums import Modality, SpeechTask, Task
from .evaluation import EvalConfig, GenerationConfig
from .model import ModelConfig, PeftConfigSpec, SpecialTokensConfig
from .rl import RewardSpec, RLConfig
from .root import GlideConfig
from .runtime import DistributedConfig, LoggingConfig
from .speech import (
    AudioAugmentConfig,
    AudioEncoderConfig,
    ProjectorConfig,
    ProjectorWarmupConfig,
    SpecAugmentConfig,
    SpeechConfig,
    SpeedPerturbConfig,
)
from .vision import VisionConfig

__all__ = [
    "AudioAugmentConfig",
    "AudioEncoderConfig",
    "DataConfig",
    "DistributedConfig",
    "EvalConfig",
    "GenerationConfig",
    "GlideConfig",
    "LoggingConfig",
    "Modality",
    "ModelConfig",
    "PackingConfig",
    "PeftConfigSpec",
    "ProjectorConfig",
    "ProjectorWarmupConfig",
    "RLConfig",
    "RewardSpec",
    "SpecAugmentConfig",
    "SpecialTokensConfig",
    "SpeechConfig",
    "SpeechTask",
    "SpeedPerturbConfig",
    "Task",
    "TemplateConfig",
    "VisionConfig",
]
