"""Validation metrics (WER / CER / BLEU / ROUGE) and the metric builder."""

from .text_metrics import (
    build_metric_fn,
    compute_bleu,
    compute_cer,
    compute_rouge,
    compute_wer,
    normalize_text,
)

__all__ = [
    "build_metric_fn",
    "compute_wer",
    "compute_cer",
    "compute_bleu",
    "compute_rouge",
    "normalize_text",
]
