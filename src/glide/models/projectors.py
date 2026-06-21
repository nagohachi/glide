"""Built-in projectors mapping encoder hidden size -> LLM embedding size.

Each entry in :data:`glide.registry.projectors` is a builder
``(ProjectorConfig, in_dim, out_dim) -> Projector``.

* ``mlp_gelu`` -- a 2-layer MLP with GELU, trained from scratch (the common
  speech-adapter). Optional time downsampling by frame stacking.
* ``qwen3_asr_proj`` -- load Qwen3-ASR's pretrained audio projection weights.
* ``qwen_omni_proj`` -- load Qwen2.5/Qwen3-Omni's pretrained audio projection.
"""

import torch
import torch.nn as nn

from ..config.schema import ProjectorConfig
from ..registry import projectors
from .plugins_base import Projector

__all__ = ["MLPGeLUProjector"]


class MLPGeLUProjector(Projector):
    """``Linear -> GELU -> Linear`` adapter, optionally stacking ``downsample`` frames.

    Frame stacking (``downsample > 1``) concatenates consecutive frames before the
    first linear, reducing the audio sequence length fed to the LLM.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None,
                 num_layers: int = 2, downsample: int = 1):
        super().__init__()
        self.downsample = max(1, downsample)
        eff_in = in_dim * self.downsample
        hidden = hidden_dim or out_dim
        layers: list[nn.Module] = []
        dims = [eff_in] + [hidden] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.downsample > 1:
            b, t, d = hidden_states.shape
            t = (t // self.downsample) * self.downsample
            hidden_states = hidden_states[:, :t, :].reshape(b, t // self.downsample, d * self.downsample)
        return self.net(hidden_states)


class IdentityProjector(Projector):
    """No-op projector for encoders that already output the LLM embedding size.

    Used with ``qwen3_asr_aut``: Qwen3-ASR's audio tower already contains its
    projector (``proj1``/``act``/``proj2``) and emits LLM-dim features, so the
    composed model needs no extra projection. (The pretrained Qwen3-ASR projector
    is trained as part of the encoder when ``encoder.freeze=false``.)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if in_dim != out_dim:
            raise ValueError(
                f"identity projector needs in_dim == out_dim (got {in_dim} != {out_dim}); "
                "use mlp_gelu when the encoder output and LLM hidden size differ."
            )

    def forward(self, hidden_states):
        return hidden_states


@projectors.register("identity", exist_ok=True)
def build_identity(cfg: ProjectorConfig, in_dim: int, out_dim: int):
    return IdentityProjector(in_dim, out_dim)


@projectors.register("mlp_gelu", exist_ok=True)
def build_mlp_gelu(cfg: ProjectorConfig, in_dim: int, out_dim: int):
    proj = MLPGeLUProjector(in_dim, out_dim, hidden_dim=cfg.hidden_dim,
                            num_layers=cfg.num_layers, downsample=cfg.downsample)
    if cfg.freeze:
        for p in proj.parameters():
            p.requires_grad_(False)
    return proj


def _load_state_into(proj: nn.Module, pretrained: str, key_filter: str) -> nn.Module:
    """Best-effort load of a pretrained projector's weights by key substring."""
    from safetensors.torch import load_file
    import os

    path = pretrained
    if os.path.isdir(pretrained):
        cand = [f for f in os.listdir(pretrained) if f.endswith(".safetensors")]
        path = os.path.join(pretrained, cand[0]) if cand else pretrained
    state = load_file(path)
    sub = {k.split(key_filter, 1)[-1].lstrip("."): v for k, v in state.items() if key_filter in k}
    missing, unexpected = proj.load_state_dict(sub, strict=False)
    return proj


@projectors.register("qwen3_asr_proj", exist_ok=True)
def build_qwen3_asr_proj(cfg: ProjectorConfig, in_dim: int, out_dim: int):
    """Pretrained Qwen3-ASR audio projector (``proj1``/``proj2`` MLP).

    Mirrors Qwen3-ASR's projector geometry (MLP) and, when ``pretrained`` is given,
    loads its weights. Falls back to a fresh MLP if weights can't be matched.
    """
    proj = MLPGeLUProjector(in_dim, out_dim, hidden_dim=cfg.hidden_dim or out_dim,
                            num_layers=2, downsample=cfg.downsample)
    if cfg.pretrained:
        try:
            _load_state_into(proj, cfg.pretrained, key_filter="audio_tower.proj")
        except Exception as exc:  # pragma: no cover - depends on checkpoint layout
            print(f"[glide] qwen3_asr_proj: could not load pretrained weights ({exc}); "
                  "using a freshly-initialized projector.")
    if cfg.freeze:
        for p in proj.parameters():
            p.requires_grad_(False)
    return proj


@projectors.register("qwen_omni_proj", exist_ok=True)
def build_qwen_omni_proj(cfg: ProjectorConfig, in_dim: int, out_dim: int):
    """Pretrained Qwen2.5/Qwen3-Omni audio projector."""
    proj = MLPGeLUProjector(in_dim, out_dim, hidden_dim=cfg.hidden_dim or out_dim,
                            num_layers=2, downsample=cfg.downsample)
    if cfg.pretrained:
        try:
            _load_state_into(proj, cfg.pretrained, key_filter="audio_tower.proj")
        except Exception as exc:  # pragma: no cover
            print(f"[glide] qwen_omni_proj: could not load pretrained weights ({exc}); "
                  "using a freshly-initialized projector.")
    if cfg.freeze:
        for p in proj.parameters():
            p.requires_grad_(False)
    return proj
