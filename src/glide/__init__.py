"""glide -- a TRL-based post-training library.

``glide`` specializes in post-training of LLMs, Speech-LLMs (speech + text input)
and Vision-LLMs (vision + text input). It supports supervised fine-tuning (SFT)
and reinforcement learning (GRPO, GSPO) on top of `TRL`_, is fully driven by
composable YAML configs, and is extended with custom plugins (audio encoders,
projectors, reward functions, metrics).

.. _TRL: https://github.com/huggingface/trl

``glide`` is a standard installable package -- install it however you like (there
is no required directory layout); plugin paths resolve against your working
directory. Typical usage from a project::

    uv pip install -e /path/to/glide   # or pip install -e, uv add, ...

    glide sft   configs/my_sft.yaml
    glide grpo  configs/my_grpo.yaml --model.name Qwen/Qwen3-1.7B

See :mod:`glide.cli.main` for the command-line entry point and
:mod:`glide.config` for the configuration system.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
