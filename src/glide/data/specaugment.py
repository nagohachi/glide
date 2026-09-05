"""SpecAugment for log-mel features (frequency + time masking).

Operates on a feature tensor shaped ``(..., n_mels, time)`` (Whisper/Qwen layout).
Masked bins are set to ``mask_value`` (0.0 by default, i.e. the log-mel floor after
normalization). Applied at train time only; a no-op when disabled.
"""

import torch

from ..config.schema import SpecAugmentConfig

__all__ = ["spec_augment"]


def spec_augment(features: torch.Tensor, cfg: SpecAugmentConfig, *,
                 attention_mask: torch.Tensor | None = None,
                 generator: torch.Generator | None = None, mask_value: float = 0.0) -> torch.Tensor:
    """Return ``features`` with SpecAugment frequency/time masks applied in-place-safe.

    Masks are drawn **independently per sample** (not once for the whole batch) so
    every utterance in a batch gets different augmentation. Time-mask widths and
    positions are scaled to each sample's **true (unpadded) length** taken from
    ``attention_mask`` -- with Whisper features padded to 3000 frames (30 s), scaling
    off the padded axis would let a single time mask zero out an entire short utterance
    (or land wholly in padding).

    Args:
        features: ``(..., n_mels, time)`` float tensor. The leading dims are flattened
            into a batch axis and masked per-sample.
        cfg: SpecAugment settings.
        attention_mask: optional ``(batch, time)`` mask of valid (non-padding) frames;
            when given, time masks are confined to each sample's valid span.
        generator: optional RNG for reproducibility.
        mask_value: value written into masked bins.
    """
    if not cfg.enabled:
        return features
    out = features.clone()
    n_mels, n_time = out.shape[-2], out.shape[-1]
    flat = out.reshape(-1, n_mels, n_time)
    batch = flat.shape[0]

    if attention_mask is not None:
        lengths = attention_mask.reshape(batch, -1).sum(dim=-1).to(torch.long).tolist()
    else:
        lengths = [n_time] * batch

    def _randint(high):
        if high <= 0:
            return 0
        return int(torch.randint(0, high, (1,), generator=generator).item())

    for b in range(batch):
        true_t = int(lengths[b]) if lengths[b] else n_time
        true_t = max(1, min(true_t, n_time))
        # Frequency masks.
        for _ in range(cfg.num_freq_mask):
            w = _randint(min(cfg.freq_mask_width, n_mels) + 1)
            if w == 0:
                continue
            f0 = _randint(n_mels - w + 1)
            flat[b, f0 : f0 + w, :] = mask_value
        # Time masks (width proportional to this utterance's true length).
        max_t = int(cfg.time_mask_width_ratio * true_t)
        for _ in range(cfg.num_time_mask):
            w = _randint(max_t + 1)
            if w == 0:
                continue
            t0 = _randint(true_t - w + 1)
            flat[b, :, t0 : t0 + w] = mask_value
    return flat.reshape(features.shape)
