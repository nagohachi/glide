"""Regression test for #8: pretrained projector loading (key remap + loud on mismatch)."""

import pytest
import torch

from glide.models.projectors import MLPGeLUProjector, _QWEN_PROJ_KEY_MAP, _load_state_into


def _save(state, tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "model.safetensors"
    save_file(state, str(path))
    return str(tmp_path)  # _load_state_into accepts a dir


def test_projector_loads_remapped_qwen_keys(tmp_path):
    proj = MLPGeLUProjector(in_dim=8, out_dim=16, hidden_dim=16, num_layers=2)
    src = {
        "audio_tower.proj1.weight": torch.randn(16, 8),
        "audio_tower.proj1.bias": torch.randn(16),
        "audio_tower.proj2.weight": torch.randn(16, 16),
        "audio_tower.proj2.bias": torch.randn(16),
    }
    _load_state_into(proj, _save(src, tmp_path), key_filter="audio_tower.proj",
                     key_map=_QWEN_PROJ_KEY_MAP)
    # Checkpoint proj1/proj2 remapped onto the projector's real net.0/net.2 params.
    assert torch.allclose(proj.net[0].weight, src["audio_tower.proj1.weight"])
    assert torch.allclose(proj.net[2].weight, src["audio_tower.proj2.weight"])


def test_projector_load_raises_on_total_mismatch(tmp_path):
    proj = MLPGeLUProjector(in_dim=8, out_dim=16, hidden_dim=16, num_layers=2)
    src = {"audio_tower.proj1.weight": torch.randn(3, 3)}  # unrelated shape
    with pytest.raises(RuntimeError):
        _load_state_into(proj, _save(src, tmp_path), key_filter="audio_tower.proj",
                         key_map=_QWEN_PROJ_KEY_MAP)
