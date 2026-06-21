"""Tests for WER/CER/BLEU/ROUGE metrics and text normalization."""

from glide.metrics import (
    build_metric_fn,
    compute_bleu,
    compute_cer,
    compute_rouge,
    compute_wer,
    normalize_text,
)


def test_normalize_text():
    assert normalize_text("Hello, WORLD!!") == "hello world"


def test_wer_perfect_and_imperfect():
    assert compute_wer(["the cat sat"], ["the cat sat"])["wer"] == 0.0
    # one substitution out of three words.
    assert abs(compute_wer(["the cat sat"], ["the dog sat"])["wer"] - 1 / 3) < 1e-6


def test_cer_zero_on_match():
    assert compute_cer(["hello"], ["hello"])["cer"] == 0.0


def test_bleu_high_on_match():
    score = compute_bleu(["the cat sat on the mat"], ["the cat sat on the mat"])["bleu"]
    assert score > 99.0


def test_rouge_keys_and_perfect():
    out = compute_rouge(["the cat sat"], ["the cat sat"])
    assert set(out) == {"rouge1", "rouge2", "rougeL"}
    assert out["rouge1"] == 1.0


def test_build_metric_fn_merges():
    fn = build_metric_fn(["wer", "cer"])
    out = fn(["hello world"], ["hello world"])
    assert out["wer"] == 0.0 and out["cer"] == 0.0
