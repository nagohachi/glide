"""Regression tests for #19: minor bug bundle (config/metrics/sampler)."""

from glide.config.schema import GlideConfig


def test_scalar_override_to_list_field_coerced():
    from glide.config.loader import dict_to_dataclass

    # `--eval.metrics cer` arrives as the bare scalar "cer"; must become ["cer"], not
    # be iterated char-by-char (KeyError: No metric named 'c').
    cfg = dict_to_dataclass(GlideConfig, {"eval": {"metrics": "cer"}})
    assert cfg.eval.metrics == ["cer"]


def test_empty_section_hydrates_to_default():
    from glide.config.loader import dict_to_dataclass

    # A YAML `training:` with everything commented out parses to None.
    cfg = dict_to_dataclass(GlideConfig, {"training": None, "model": None})
    assert cfg.training == {}
    assert isinstance(cfg.model.name, str)


def test_sampler_global_length_sort_when_not_shuffling():
    from glide.data.sampler import LengthGroupedBatchSampler

    s = LengthGroupedBatchSampler(
        [5, 1, 4, 2, 3], batch_size=2, mega_batch_mult=1, shuffle=False
    )
    # First batch holds the two globally-longest samples (ids 0, 2), not a
    # per-megabatch-local pair like [0, 1].
    assert list(s)[0] == [0, 2]


def test_build_metric_fn_bleu_not_normalized_and_single_call():
    from glide.metrics.text_metrics import build_metric_fn
    from glide.registry import metrics

    original_bleu = metrics.get("bleu")
    seen = []

    def _fake_bleu(preds, refs, *, normalize=False):
        seen.append(normalize)
        return {"bleu": 0.0}

    metrics.register("bleu", _fake_bleu, exist_ok=True)
    try:
        build_metric_fn(["bleu"], normalize=True)(["a"], ["a"])
    finally:
        metrics.register("bleu", original_bleu, exist_ok=True)
    # BLEU is not run through glide's normalizer, and is invoked exactly once
    # (no `except TypeError` double-invoke).
    assert seen == [False]
