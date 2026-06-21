"""Text metrics for validation-time decoding: WER, CER, BLEU, ROUGE.

Each metric is a callable ``(predictions, references) -> dict[str, float]`` and is
registered in :data:`glide.registry.metrics` under its name so it can be selected
from YAML (``eval.metrics: [wer, cer]``) or extended by plugins.

* ``wer`` / ``cer`` -- word/character error rate via ``jiwer`` (speech recognition).
* ``bleu`` -- corpus BLEU via ``sacrebleu`` (translation).
* ``rouge`` -- ROUGE-1/2/L F-measure via ``rouge_score`` (translation/summarization).
"""

import re
import unicodedata
from typing import Sequence

from ..registry import metrics

__all__ = ["normalize_text", "compute_wer", "compute_cer", "compute_bleu", "compute_rouge",
           "build_metric_fn"]


def normalize_text(text: str) -> str:
    """Lowercase, NFKC-normalize, strip punctuation and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _prep(preds, refs, normalize):
    preds = [normalize_text(p) if normalize else p for p in preds]
    refs = [normalize_text(r) if normalize else r for r in refs]
    return preds, refs


@metrics.register("wer")
def compute_wer(predictions: Sequence[str], references: Sequence[str], *, normalize=True):
    """Word Error Rate (lower is better)."""
    import jiwer

    preds, refs = _prep(list(predictions), list(references), normalize)
    # jiwer errors on empty references; guard each.
    refs = [r if r else " " for r in refs]
    return {"wer": float(jiwer.wer(refs, preds))}


@metrics.register("cer")
def compute_cer(predictions: Sequence[str], references: Sequence[str], *, normalize=True):
    """Character Error Rate (lower is better)."""
    import jiwer

    preds, refs = _prep(list(predictions), list(references), normalize)
    refs = [r if r else " " for r in refs]
    return {"cer": float(jiwer.cer(refs, preds))}


@metrics.register("bleu")
def compute_bleu(predictions: Sequence[str], references: Sequence[str], *, normalize=False):
    """Corpus BLEU (higher is better)."""
    import sacrebleu

    preds, refs = _prep(list(predictions), list(references), normalize)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return {"bleu": float(bleu.score)}


@metrics.register("rouge")
def compute_rouge(predictions: Sequence[str], references: Sequence[str], *, normalize=True):
    """ROUGE-1/2/L F-measure averaged over the corpus (higher is better)."""
    from rouge_score import rouge_scorer

    preds, refs = _prep(list(predictions), list(references), normalize)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = max(1, len(preds))
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        for k in totals:
            totals[k] += scores[k].fmeasure
    return {k: v / n for k, v in totals.items()}


def build_metric_fn(names: Sequence[str], *, normalize: bool = True):
    """Return a single callable computing all requested metrics.

    Args:
        names: Metric names resolved from :data:`glide.registry.metrics`.
        normalize: Passed to each metric (text normalization).

    Returns:
        ``(predictions, references) -> dict[str, float]`` merging every metric.
    """
    fns = [(name, metrics.get(name)) for name in names]

    def _compute(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, fn in fns:
            try:
                out.update(fn(predictions, references, normalize=normalize))
            except TypeError:
                out.update(fn(predictions, references))
        return out

    return _compute
