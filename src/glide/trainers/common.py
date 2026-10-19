"""Shared helpers for building trainers: plugins, rewards, PEFT and generation eval."""

from typing import Callable

from transformers import TrainerCallback

from ..config.schema import GlideConfig
from ..data.jsonl import read_jsonl_files
from ..eval.generate import GenerateEvalCallback, GenerationEvaluator, TestEvalCallback
from ..registry import load_plugins, rewards

__all__ = ["init_plugins", "build_reward_funcs", "build_peft_config", "build_eval_callbacks"]


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
        build_reward = rewards.get(spec.name)
        fn = build_reward(**spec.kwargs)
        # Name the reward by its spec name so TRL logs distinct rewards/{name}/mean
        # series. Builders return a closure that shares one name across rewards
        # (``_compute_reward``), so keeping fn.__name__ would collapse every reward
        # into one colliding metric key.
        fn.__name__ = spec.name
        funcs.append(fn)
        weights.append(spec.weight)
    return funcs, weights


def build_peft_config(config: GlideConfig):
    """Return the ``peft.LoraConfig`` for ``config.peft``, or ``None`` when disabled."""
    if not config.peft.enabled:
        return None
    from peft import LoraConfig

    p = config.peft
    target_modules = p.target_modules
    modules_to_save = list(p.modules_to_save or [])
    if config.is_composed_speech_llm():
        # Composed SpeechLLM is a plain nn.Module (not a PreTrainedModel). 'all-linear'
        # would LoRA-wrap EVERY nn.Linear in the tree: the from-scratch projector (its
        # random base weights frozen -> the modality bridge can never learn), the frozen
        # encoder's linears (unintended trainable adapters), and lm_head. Instead target
        # only the LLM's attention/MLP projections via a regex anchored on `llm.` (so the
        # encoder's identically-named q_proj/... are excluded), and keep the projector
        # fully trainable through modules_to_save.
        if target_modules == "all-linear" or target_modules is None:
            target_modules = (
                r"llm\.model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
            )
        if "projector" not in modules_to_save:
            modules_to_save.append("projector")
    return LoraConfig(
        r=p.r,
        lora_alpha=p.lora_alpha,
        lora_dropout=p.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save or None,
        bias=p.bias,
        task_type="CAUSAL_LM",
    )


def build_eval_callbacks(config: GlideConfig, processor) -> list[TrainerCallback]:
    """Return the AR-decoding callbacks ``config.eval.generate`` asks for.

    * :class:`GenerateEvalCallback` over ``data.eval_jsonl_path`` at every evaluation,
      capped at ``data.max_eval_samples`` (generation is slow);
    * :class:`TestEvalCallback` over all of ``data.test_jsonl_path`` at the end of
      training (never capped).

    Each callback still needs :meth:`bind_trainer` once the trainer exists, so its
    metrics are logged through the trainer.
    """
    if not config.eval.generate.enabled:
        return []
    callbacks: list[TrainerCallback] = []
    eval_records = read_jsonl_files(config.data.eval_jsonl_path)
    if eval_records:
        callbacks.append(GenerateEvalCallback(GenerationEvaluator(config, processor, eval_records)))
    test_records = read_jsonl_files(config.data.test_jsonl_path)
    if test_records:
        evaluator = GenerationEvaluator(config, processor, test_records, cap_samples=False)
        callbacks.append(TestEvalCallback(evaluator))
    return callbacks
