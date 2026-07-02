"""Shared helpers for building trainers: rewards, generation eval, plugin loading."""

from typing import Callable

from ..config.schema import GlideConfig
from ..eval.generate import GenerateEvalCallback, GenerationEvaluator, TestEvalCallback
from ..registry import load_plugins, rewards

__all__ = ["init_plugins", "build_reward_funcs", "maybe_generation_callback", "maybe_test_callback"]


def init_plugins(config: GlideConfig) -> None:
    """Import built-in plugins and any user plugins named in the config."""
    import glide.plugins  # noqa: F401  registers built-in rewards/metrics

    if config.plugins:
        load_plugins(config.plugins)


def build_reward_funcs(config: GlideConfig) -> tuple[list[Callable], list[float]]:
    """Resolve reward builders from the registry into ``(funcs, weights)``."""
    funcs: list[Callable] = []
    weights: list[float] = []
    for spec in config.rl.rewards:
        builder = rewards.get(spec.name)
        fn = builder(**spec.kwargs)
        fn.__name__ = getattr(fn, "__name__", spec.name) or spec.name
        funcs.append(fn)
        weights.append(spec.weight)
    return funcs, weights


def maybe_generation_callback(config: GlideConfig, processor, eval_records):
    """Build a :class:`GenerateEvalCallback` if AR-decoding eval is enabled."""
    if not config.eval.generate.enabled or not eval_records:
        return None
    evaluator = GenerationEvaluator(config, processor, eval_records)
    return GenerateEvalCallback(evaluator)


def maybe_test_callback(config: GlideConfig, processor):
    """Build a :class:`TestEvalCallback` from ``data.test`` if configured."""
    if not config.eval.generate.enabled or config.data.test is None:
        return None
    from ..data.jsonl import read_jsonl

    paths = config.data.test if isinstance(config.data.test, list) else [config.data.test]
    test_records = []
    for p in paths:
        test_records.extend(read_jsonl(p))
    if not test_records:
        return None
    # Test evaluation always scores the full set; max_eval_samples only caps validation.
    evaluator = GenerationEvaluator(config, processor, test_records, cap_samples=False)
    return TestEvalCallback(evaluator)
