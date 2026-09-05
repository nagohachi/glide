"""Enumerated choices used across the schema."""

from enum import StrEnum

__all__ = ["Task", "Modality", "SpeechTask"]


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
