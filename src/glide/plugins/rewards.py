"""Built-in reward functions for RL (GRPO / GSPO).

Each entry in :data:`glide.registry.rewards` is a *builder*: a callable that takes
the ``kwargs`` from a :class:`~glide.config.schema.RewardSpec` and returns the
actual reward function. A reward function follows TRL's contract::

    def reward(prompts, completions, completion_ids=None, **columns) -> list[float]

where ``columns`` are the extra dataset columns (e.g. ``reference``), one value
per sample. To add your own, register a builder in your plugin module::

    from glide.registry import rewards

    @rewards.register("my_reward")
    def build_my_reward(scale: float = 1.0):
        def _reward(prompts, completions, **kw):
            return [scale * f(c) for c in completions]
        return _reward
"""

import re
from typing import Any, Callable

from ..registry import rewards

__all__ = ["completion_text", "build_format_reward", "build_length_reward",
           "build_exact_match_reward", "build_cer_reward"]


def completion_text(completion: Any) -> str:
    """Extract the assistant text from a TRL completion (str or message list)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):  # conversational: list of message dicts
        parts = []
        for msg in completion:
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            parts.append(content)
        return " ".join(parts)
    return str(completion)


@rewards.register("format")
def build_format_reward(pattern: str = r"<think>.*?</think>", flags: str = "s") -> Callable:
    """Reward 1.0 when the completion matches ``pattern`` (default: a ``<think>`` block).

    Args:
        pattern: A regular expression searched in the completion text.
        flags: Any of ``"s"`` (dotall), ``"i"`` (ignorecase), ``"m"`` (multiline).
    """
    re_flags = 0
    re_flags |= re.S if "s" in flags else 0
    re_flags |= re.I if "i" in flags else 0
    re_flags |= re.M if "m" in flags else 0
    compiled = re.compile(pattern, re_flags)

    def _reward(prompts=None, completions=None, **kwargs):
        completions = completions or []
        return [1.0 if compiled.search(completion_text(c)) else 0.0 for c in completions]

    return _reward


@rewards.register("length")
def build_length_reward(target: int = 256, scale: float = 0.001) -> Callable:
    """Reward proximity of completion length (chars) to ``target`` (mild shaping)."""

    def _reward(prompts=None, completions=None, **kwargs):
        completions = completions or []
        return [
            max(0.0, 1.0 - scale * abs(len(completion_text(c)) - target))
            for c in completions
        ]

    return _reward


@rewards.register("exact_match")
def build_exact_match_reward(reference_key: str = "reference", normalize: bool = True) -> Callable:
    """Reward 1.0 when the completion exactly matches the reference column."""
    from ..metrics.text_metrics import normalize_text

    def _reward(prompts=None, completions=None, **kwargs):
        completions = completions or []
        refs = kwargs.get(reference_key)
        if refs is None:
            return [0.0] * len(completions)
        out = []
        for c, r in zip(completions, refs):
            pred = completion_text(c)
            if normalize:
                pred, r = normalize_text(pred), normalize_text(r or "")
            out.append(1.0 if pred == r else 0.0)
        return out

    return _reward


@rewards.register("cer")
def build_cer_reward(reference_key: str = "reference", normalize: bool = True) -> Callable:
    """Reward ``1 - CER`` between completion and the reference column (ASR-style)."""
    import jiwer

    from ..metrics.text_metrics import normalize_text

    def _reward(prompts=None, completions=None, **kwargs):
        completions = completions or []
        refs = kwargs.get(reference_key)
        if refs is None:
            return [0.0] * len(completions)
        out = []
        for c, r in zip(completions, refs):
            pred, ref = completion_text(c), (r or "")
            if normalize:
                pred, ref = normalize_text(pred), normalize_text(ref)
            if not ref:
                out.append(0.0)
                continue
            cer = jiwer.cer(ref, pred if pred else " ")
            out.append(max(0.0, 1.0 - float(cer)))
        return out

    return _reward
