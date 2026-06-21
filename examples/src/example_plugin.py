"""Example project plugin.

Demonstrates the project-plugin workflow: with ``glide`` cloned into ``libs/`` and
editable installed, drop files like this under your project ``src/`` and reference
them from the config::

    plugins: ["examples/src/example_plugin.py"]
    rl:
      rewards:
        - { name: keyword_reward, weight: 1.0, kwargs: { keyword: "answer" } }

No ``sys.path`` manipulation is needed -- ``glide`` imports the file directly.
"""

from glide.plugins.rewards import completion_text
from glide.registry import metrics, rewards


@rewards.register("keyword_reward")
def build_keyword_reward(keyword: str = "answer"):
    """Reward 1.0 when the completion contains ``keyword`` (case-insensitive)."""

    def _reward(prompts=None, completions=None, **columns):
        completions = completions or []
        kw = keyword.lower()
        return [1.0 if kw in completion_text(c).lower() else 0.0 for c in completions]

    return _reward


@metrics.register("exact_match_rate")
def exact_match_rate(predictions, references, *, normalize=True):
    """Fraction of predictions exactly equal to their reference."""
    from glide.metrics import normalize_text

    n = max(1, len(predictions))
    hits = 0
    for p, r in zip(predictions, references):
        if normalize:
            p, r = normalize_text(p), normalize_text(r)
        hits += int(p == r)
    return {"exact_match_rate": hits / n}
