"""glide — a TRL-based post-training library.

``glide`` specializes in post-training of LLMs, Speech-LLMs (speech + text input)
and Vision-LLMs (vision + text input). It supports supervised fine-tuning (SFT)
and reinforcement learning (GRPO, GSPO) on top of `TRL`_, is fully driven by
composable YAML configs, and is extended with custom plugins (audio encoders,
projectors, reward functions, metrics).

.. _TRL: https://github.com/huggingface/trl

``glide`` is a standard installable package — install it however you like (there
is no required directory layout); plugin paths resolve against your working
directory. Typical usage from a project::

    uv pip install -e /path/to/glide   # or pip install -e, uv add, ...

    glide sft   configs/my_sft.yaml
    glide grpo  configs/my_grpo.yaml --model.model_name_or_id Qwen/Qwen3-1.7B

Module overview
---------------

:mod:`glide.config`
    Typed configuration schema (:class:`~glide.config.schema.GlideConfig` and
    friends) and the YAML + CLI override loader. Start here to understand every
    knob exposed by a run YAML.

:mod:`glide.trainers`
    High-level trainer builders (:func:`~glide.trainers.build_trainer`) that wire
    together TRL, the dataset, callbacks and distributed launch for SFT, GRPO and
    GSPO.

:mod:`glide.eval`
    Validation-time autoregressive decoding (:class:`~glide.eval.GenerationEvaluator`)
    and the Trainer callback (:class:`~glide.eval.GenerateEvalCallback`) that plugs
    it into the training loop.

:mod:`glide.data`
    JSONL loading, chat-template application, dataset builders and collators for
    text, speech and vision modalities.

:mod:`glide.models`
    Model + processor loading (:func:`~glide.models.load_model_and_processor`),
    the composable Speech-LLM architecture, and the built-in encoder / projector
    registry.

:mod:`glide.metrics`
    Text metric functions (WER, CER, BLEU, ROUGE) and
    :func:`~glide.metrics.build_metric_fn`, the factory used at eval time.

:mod:`glide.plugins`
    Plugin-loading entry point and built-in reward functions. Custom plugins
    (encoders, projectors, reward functions, metrics) are registered here.

:mod:`glide.cli.main`
    Command-line entry point for ``glide sft``, ``glide grpo``, ``glide eval``
    and ``glide docs``.
"""

__version__ = "0.1.0"

from . import cli, config, data, eval, metrics, models, plugins, trainers

__all__ = ["__version__", "cli", "config", "data", "eval", "metrics", "models", "plugins", "trainers"]
