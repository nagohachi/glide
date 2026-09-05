"""Tests for composed Speech-LLM building blocks (CPU, no model downloads)."""

import numpy as np
import torch

from glide.config.schema import (
    GlideConfig,
    AudioEncoderConfig,
    ProjectorConfig,
    SpecAugmentConfig,
    SpeedPerturbConfig,
)
from glide.data.audio import speed_perturb
from glide.data.specaugment import spec_augment
from glide.models.projectors import MLPGeLUProjector


# --- projector ---------------------------------------------------------------
def test_mlp_gelu_projector_maps_dims():
    proj = MLPGeLUProjector(in_dim=128, out_dim=64, hidden_dim=256, num_layers=2)
    out = proj(torch.randn(2, 10, 128))
    assert out.shape == (2, 10, 64)


def test_mlp_gelu_projector_downsample_stacks_frames():
    proj = MLPGeLUProjector(in_dim=32, out_dim=16, downsample=4)
    out = proj(torch.randn(1, 20, 32))   # 20 frames / 4 -> 5
    assert out.shape == (1, 5, 16)


# --- specaugment -------------------------------------------------------------
def test_spec_augment_disabled_is_noop():
    feats = torch.randn(1, 80, 100)
    same = spec_augment(feats, SpecAugmentConfig(enabled=False))
    assert torch.equal(feats, same)


def test_spec_augment_masks_some_bins_and_keeps_shape():
    feats = torch.ones(1, 80, 200)
    g = torch.Generator().manual_seed(0)
    out = spec_augment(
        feats, SpecAugmentConfig(enabled=True, num_freq_mask=2, num_time_mask=2), generator=g
    )
    assert out.shape == feats.shape
    assert (out == 0.0).any()          # something got masked
    assert not torch.equal(out, feats)


# --- speed perturbation ------------------------------------------------------
def test_speed_perturb_identity_at_factor_one():
    wav = np.random.randn(16000).astype(np.float32)
    assert np.array_equal(speed_perturb(wav, 1.0, 16000), wav)


def test_speed_perturb_changes_length():
    wav = np.random.randn(16000).astype(np.float32)
    faster = speed_perturb(wav, 1.5, 16000)   # ~1/1.5 the samples
    assert abs(len(faster) - 16000 / 1.5) < 50


# --- config hydration of the nested speech schema ----------------------------
def test_speech_config_nested_hydration():
    from glide.config.loader import dict_to_dataclass

    cfg = dict_to_dataclass(GlideConfig, {
        "modality": "speech",
        "speech": {
            "encoder": {"name": "whisper", "pretrained": "openai/whisper-small", "freeze": True},
            "projector": {"name": "mlp_gelu", "downsample": 5},
            "augment": {
                "specaugment": {"enabled": True, "num_time_mask": 3},
                "speed_perturb": {"enabled": True, "from_field": True},
            },
        },
    })
    assert isinstance(cfg.speech.encoder, AudioEncoderConfig)
    assert cfg.speech.encoder.name == "whisper" and cfg.speech.encoder.freeze
    assert isinstance(cfg.speech.projector, ProjectorConfig)
    assert cfg.speech.projector.downsample == 5
    assert isinstance(cfg.speech.augment.specaugment, SpecAugmentConfig)
    assert cfg.speech.augment.specaugment.num_time_mask == 3
    assert isinstance(cfg.speech.augment.speed_perturb, SpeedPerturbConfig)
    assert cfg.speech.augment.speed_perturb.from_field


def test_builtin_encoders_and_projectors_registered():
    import glide.models  # noqa: F401  triggers registration
    from glide.registry import audio_encoders, projectors

    for name in ("whisper", "wavlm", "xls_r", "qwen3_asr_aut", "qwen_omni_aut"):
        assert name in audio_encoders
    for name in ("mlp_gelu", "qwen3_asr_proj", "qwen_omni_proj"):
        assert name in projectors
