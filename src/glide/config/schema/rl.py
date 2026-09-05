"""Reinforcement-learning (GRPO / GSPO) settings."""

from dataclasses import dataclass, field
from typing import Any

__all__ = ["RewardSpec", "RLConfig"]


@dataclass
class RewardSpec:
    """A single reward function entry for RL training.

    ``name`` resolves a function in the reward registry (built-in or plugin).
    ``weight`` scales its contribution; ``kwargs`` are passed at construction.
    """

    name: str
    weight: float = 1.0
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RLConfig:
    """Reinforcement-learning settings (GRPO / GSPO)."""

    #: Reward functions (composed as a weighted sum). Required for GRPO/GSPO.
    rewards: list[RewardSpec] = field(default_factory=list)
    #: Use a vLLM rollout server for generation (``server`` / ``colocate``).
    use_vllm: bool = False
    vllm_mode: str = "server"
    #: Prefill string prepended to every generation (e.g. a CoT opener).
    response_prefix: str | None = None

    # --- Speech GSPO loop (glide.trainers.rl_speech) ---------------------- #
    # These drive the self-contained speech-in-the-loop GSPO/GRPO trainer, which
    # generates with audio (text-only TRL GRPO can't) and rolls out through a vLLM
    # server on a *separate* GPU from training. Ignored by the text TRL path.
    #: Number of sampled completions per prompt (the GRPO/GSPO group size G).
    num_generations: int = 8
    #: Rollout sampling temperature / nucleus cutoff.
    temperature: float = 1.0
    top_p: float = 1.0
    #: Max new tokens per rollout completion.
    max_completion_length: int = 256
    #: Clip range eps for the (sequence-level, GSPO) policy ratio.
    clip_eps: float = 0.2
    #: KL-to-reference penalty coefficient (0 disables the reference model entirely).
    kl_beta: float = 0.0
    #: Group-normalise advantages by their std (GRPO ``scale_rewards``).
    scale_rewards: bool = True
    #: Inner policy updates per rollout batch (mu). 1 == on-policy (ratio == 1).
    num_iterations: int = 1
    #: vLLM rollout server address (``server`` mode). The server runs the *same*
    #: composed model on a different GPU; weights are pushed after each update.
    vllm_server_host: str = "localhost"
    vllm_server_port: int = 8000
    #: Refresh the vLLM server weights every N optimizer steps (1 == every step).
    vllm_sync_every: int = 1
    #: Shared path where the trainer writes the remapped policy weights for the vLLM
    #: server to reload (weight sync). Must be readable by the server host.
    vllm_sync_path: str = "outputs/vllm_sync_policy.safetensors"
    #: Reward field on each record consumed by the CER reward (the reference text).
    reference_key: str = "reference"
