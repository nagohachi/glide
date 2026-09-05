"""Tests for model-loading utilities (dtype resolution, generation-config fix)."""

import types

import torch

from glide.models.loader import _sanitize_generation_config, resolve_dtype


def test_resolve_dtype():
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype("fp16") is torch.float16
    assert resolve_dtype("float32") is torch.float32
    assert resolve_dtype("auto") == "auto"


def _model_with_genconfig(**kw):
    return types.SimpleNamespace(generation_config=types.SimpleNamespace(**kw))


def test_sanitize_resets_sampling_flags_when_greedy():
    # Mirrors Qwen3-ASR's shipped config: do_sample=False but temperature set,
    # which recent transformers refuse to save.
    m = _model_with_genconfig(do_sample=False, temperature=1e-6, top_p=0.8, top_k=20)
    _sanitize_generation_config(m)
    assert m.generation_config.temperature == 1.0
    assert m.generation_config.top_p == 1.0
    assert m.generation_config.top_k == 50


def test_sanitize_leaves_sampling_config_untouched():
    m = _model_with_genconfig(do_sample=True, temperature=0.7, top_p=0.8)
    _sanitize_generation_config(m)
    assert m.generation_config.temperature == 0.7  # sampling run -> keep as-is
    assert m.generation_config.top_p == 0.8


def test_sanitize_no_generation_config_is_noop():
    _sanitize_generation_config(types.SimpleNamespace(generation_config=None))  # no error
