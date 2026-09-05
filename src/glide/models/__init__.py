"""Model loading, special tokens, and composable Speech-LLM components."""

from . import encoders, projectors  # noqa: F401  (register built-in builders)
from .encoders import WaveformEncoder, WhisperEncoder
from .loader import LoadedModel, load_model_and_processor, resolve_dtype
from .plugins_base import AudioEncoder, Projector
from .projectors import MLPGeLUProjector
from .special_tokens import SpecialTokenInfo, apply_special_tokens
from .speech_llm import SpeechLLM, build_speech_llm

__all__ = [
    "LoadedModel",
    "load_model_and_processor",
    "resolve_dtype",
    "apply_special_tokens",
    "SpecialTokenInfo",
    "AudioEncoder",
    "Projector",
    "WhisperEncoder",
    "WaveformEncoder",
    "MLPGeLUProjector",
    "SpeechLLM",
    "build_speech_llm",
]
