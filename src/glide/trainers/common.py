"""Shared helpers for building trainers: rewards, generation eval, plugin loading."""

from typing import Callable

from ..config.schema import GlideConfig
from ..eval.generate import GenerateEvalCallback, GenerationEvaluator
from ..registry import load_plugins, rewards

__all__ = ["init_plugins", "build_reward_funcs", "maybe_generation_callback"]


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
