"""Validation decoding and metrics."""

from dataclasses import dataclass, field

__all__ = ["GenerationConfig", "EvalConfig"]


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
