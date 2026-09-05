"""Reinforcement-learning trainer construction (GRPO / GSPO).

* **GRPO** -> TRL ``GRPOTrainer`` with token-level importance sampling.
* **GSPO** -> the same ``GRPOTrainer`` with ``importance_sampling_level="sequence"``
  (Group *Sequence* Policy Optimization is GRPO with sequence-level IS).

Reward functions and their weights come from the config (see
:func:`glide.trainers.common.build_reward_funcs`).

* **text** modality -> the TRL ``GRPOTrainer`` path above.
* **speech** modality -> :class:`glide.trainers.rl_speech.SpeechGSPOTrainer`, a
  self-contained loop that rolls out *with audio* (which the text TRL path cannot)
  through a vLLM server on a separate GPU. Vision RL is still gated with an error.
"""

from ..config.schema import GlideConfig, Modality, Task
from ..config.loader import build_training_args
from ..data.build import build_rl_text_dataset
from ..models.loader import load_model_and_processor
from .common import build_reward_funcs, init_plugins
from .sft import _peft_config

__all__ = ["build_rl_trainer"]


def _check_modality(config: GlideConfig) -> None:
    if config.modality is Modality.VISION:
        raise NotImplementedError(
            f"RL ({config.task.value}) does not support the vision modality yet. "
            "Text and speech (audio-in-the-loop GSPO/GRPO) are supported."
        )


def build_rl_trainer(config: GlideConfig):
    """Build a ready-to-train RL trainer (GRPO/GSPO) from ``config``."""
    init_plugins(config)
    _check_modality(config)

    if config.task not in (Task.GRPO, Task.GSPO):
        raise ValueError(f"Unsupported RL task: {config.task}")

    # Speech RL needs audio in the rollout -> dedicated loop (TRL GRPO is text-only).
    if config.modality is Modality.SPEECH:
        from .rl_speech import build_speech_rl_trainer

        return build_speech_rl_trainer(config)

    loaded = load_model_and_processor(config)
    reward_funcs, reward_weights = build_reward_funcs(config)
    return _build_grpo_trainer(config, loaded, reward_funcs, reward_weights)


def _common_rl_training_defaults(config: GlideConfig) -> None:
    t = config.training
    if config.rl.use_vllm:
        t.setdefault("use_vllm", True)
        t.setdefault("vllm_mode", config.rl.vllm_mode)


def _build_grpo_trainer(config, loaded, reward_funcs, reward_weights):
    from trl import GRPOConfig, GRPOTrainer

    _common_rl_training_defaults(config)
    t = config.training
    if reward_weights:
        t.setdefault("reward_weights", reward_weights)
    # GSPO == GRPO with sequence-level importance sampling.
    if config.task is Task.GSPO:
        t.setdefault("importance_sampling_level", "sequence")

    args = build_training_args(config, GRPOConfig)
    return GRPOTrainer(
        model=loaded.model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=build_rl_text_dataset(config, "train"),
        eval_dataset=build_rl_text_dataset(config, "eval"),
        processing_class=loaded.tokenizer,
        peft_config=_peft_config(config),
    )
