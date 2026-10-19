"""Built-in projectors mapping encoder hidden size -> LLM embedding size.

Each entry in :data:`glide.registry.projectors` is a builder
``(ProjectorConfig, in_dim, out_dim) -> Projector``.

* ``mlp_gelu`` -- a 2-layer MLP with GELU, trained from scratch (the common
  speech-adapter). Optional time downsampling by frame stacking.
* ``qwen3_asr_proj`` -- load Qwen3-ASR's pretrained audio projection weights.
* ``qwen_omni_proj`` -- load Qwen2.5/Qwen3-Omni's pretrained audio projection.
"""

from typing import Any

import torch
import torch.nn as nn

from ..config.schema import ProjectorConfig
from ..registry import projectors
from .plugins_base import Projector

from ..logging_utils import get_logger

_log = get_logger("models")

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
    # IdentityProjector cannot downsample (frame stacking would change the feature dim),
    # yet SpeechLLM still divides audio_lengths by projector.downsample -> the spliced
    # audio would be silently truncated. Reject the incompatible combination loudly.
    if cfg.downsample and cfg.downsample > 1:
        raise ValueError(
            f"projector 'identity' does not support downsample>1 (got {cfg.downsample}); "
            "SpeechLLM would divide audio_lengths without shortening the frames, "
            "truncating the spliced audio. Use 'mlp_gelu' for frame stacking."
        )
    return IdentityProjector(in_dim, out_dim)


@projectors.register("mlp_gelu", exist_ok=True)
def build_mlp_gelu(cfg: ProjectorConfig, in_dim: int, out_dim: int):
    proj = MLPGeLUProjector(in_dim, out_dim, hidden_dim=cfg.hidden_dim,
                            num_layers=cfg.num_layers, downsample=cfg.downsample)
    if cfg.freeze:
        for p in proj.parameters():
            p.requires_grad_(False)
    return proj


def _resolve_pretrained_file(pretrained: str) -> str:
    """Resolve ``pretrained`` (local file, local dir, or HF hub id) to a weights file."""
    import os

    if os.path.isfile(pretrained):
        return pretrained
    if not os.path.isdir(pretrained):
        # Treat as an HF hub id and pull a local snapshot (schema documents "HF id / path").
        from huggingface_hub import snapshot_download

        pretrained = snapshot_download(pretrained, allow_patterns=["*.safetensors", "*.json"])
    cand = sorted(f for f in os.listdir(pretrained) if f.endswith(".safetensors"))
    if not cand:
        raise FileNotFoundError(f"no .safetensors found under {pretrained!r}")
    return os.path.join(pretrained, cand[0])


def _load_state_into(
    proj: nn.Module, pretrained: str, key_filter: str, key_map: dict[str, str] | None = None
) -> nn.Module:
    """Load a pretrained projector's weights by key substring, remapping to real names.

    Checkpoint keys under ``key_filter`` (e.g. ``audio_tower.proj1.weight``) are stripped
    to their tail (``proj1.weight``) and, via ``key_map``, remapped to the projector's own
    parameter names (``net.0.weight``). Raises loudly if nothing usable was matched -- the
    old code discarded ``load_state_dict``'s missing/unexpected results, so a total mismatch
    silently trained from random init while the user believed weights were warm-started.
    """
    from safetensors.torch import load_file

    path = _resolve_pretrained_file(pretrained)
    state = load_file(path)
    sub: dict[str, Any] = {}
    for k, v in state.items():
        if key_filter not in k:
            continue
        tail = k.split(key_filter, 1)[-1].lstrip(".")
        sub[(key_map or {}).get(tail, tail)] = v

    missing, unexpected = proj.load_state_dict(sub, strict=False)
    loaded = [k for k in sub if k not in set(unexpected)]
    if not loaded or missing:
        raise RuntimeError(
            f"pretrained projector load from {pretrained!r} matched "
            f"{len(loaded)}/{len(proj.state_dict())} params "
            f"(missing={list(missing)}, unexpected={list(unexpected)}). "
            f"Projector expects keys {list(proj.state_dict())}; got source tails "
            f"{[k.split(key_filter, 1)[-1].lstrip('.') for k in state if key_filter in k]}."
        )
    return proj


# Qwen audio-projector checkpoint tails -> MLPGeLUProjector param names.
# net.0 = first Linear, net.1 = GELU, net.2 = second Linear.
_QWEN_PROJ_KEY_MAP = {
    "proj1.weight": "net.0.weight", "proj1.bias": "net.0.bias",
    "proj2.weight": "net.2.weight", "proj2.bias": "net.2.bias",
    "1.weight": "net.0.weight", "1.bias": "net.0.bias",
    "2.weight": "net.2.weight", "2.bias": "net.2.bias",
}


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
            _load_state_into(proj, cfg.pretrained, key_filter="audio_tower.proj",
                             key_map=_QWEN_PROJ_KEY_MAP)
        except Exception as exc:  # pragma: no cover - depends on checkpoint layout
            _log.warning(
                "%s: could not load pretrained weights (%s); "
                "using a freshly-initialized projector.", "qwen3_asr_proj", exc
            )
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
            _load_state_into(proj, cfg.pretrained, key_filter="audio_tower.proj",
                             key_map=_QWEN_PROJ_KEY_MAP)
        except Exception as exc:  # pragma: no cover
            _log.warning(
                "%s: could not load pretrained weights (%s); "
                "using a freshly-initialized projector.", "qwen_omni_proj", exc
            )
    if cfg.freeze:
        for p in proj.parameters():
            p.requires_grad_(False)
    return proj
