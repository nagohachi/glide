"""SpecAugment for log-mel features (frequency + time masking).

Operates on a feature tensor shaped ``(..., n_mels, time)`` (Whisper/Qwen layout).
Masked bins are set to ``mask_value`` (0.0 by default, i.e. the log-mel floor after
normalization). Applied at train time only; a no-op when disabled.
"""

import torch

from ..config.schema import SpecAugmentConfig

__all__ = ["spec_augment"]


def spec_augment(features: torch.Tensor, cfg: SpecAugmentConfig, *,
                 generator: torch.Generator | None = None, mask_value: float = 0.0) -> torch.Tensor:
    """Return ``features`` with SpecAugment frequency/time masks applied in-place-safe.

    Args:
        features: ``(..., n_mels, time)`` float tensor.
        cfg: SpecAugment settings.
        generator: optional RNG for reproducibility.
        mask_value: value written into masked bins.
    """
    if not cfg.enabled:
        return features
    out = features.clone()
    n_mels, n_time = out.shape[-2], out.shape[-1]

    def _randint(high):
        if high <= 0:
            return 0
        return int(torch.randint(0, high, (1,), generator=generator).item())

    # Frequency masks.
    for _ in range(cfg.num_freq_mask):
        w = _randint(min(cfg.freq_mask_width, n_mels) + 1)
        if w == 0:
            continue
        f0 = _randint(n_mels - w + 1)
        out[..., f0 : f0 + w, :] = mask_value

    # Time masks (width proportional to utterance length).
    max_t = int(cfg.time_mask_width_ratio * n_time)
    for _ in range(cfg.num_time_mask):
        w = _randint(max_t + 1)
        if w == 0:
            continue
        t0 = _randint(n_time - w + 1)
        out[..., :, t0 : t0 + w] = mask_value
    return out
