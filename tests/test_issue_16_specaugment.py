"""Regression test for #16: SpecAugment per-sample masks scaled to true length."""

import torch

from glide.config.schema import SpecAugmentConfig


def test_specaugment_per_sample_and_length_scaled():
    from glide.data.specaugment import spec_augment

    cfg = SpecAugmentConfig(enabled=True, freq_mask_width=2, num_freq_mask=1,
                            time_mask_width_ratio=0.3, num_time_mask=3)
    b, mels, time = 4, 8, 100
    feats = torch.ones(b, mels, time)
    # Sample 0 is very short (true length 10); others are full length.
    attn = torch.ones(b, time)
    attn[0, 10:] = 0
    gen = torch.Generator().manual_seed(0)
    out = spec_augment(feats, cfg, attention_mask=attn, generator=gen)

    # Per-sample masks differ (batch-wide masking would make all rows identical).
    assert not torch.equal(out[0], out[1])

    # For the short sample, any fully-zeroed column (a time mask) stays within its
    # true length -- it never lands in the padding region [10, 100).
    fully_zero_cols = (out[0] == 0).all(dim=0).nonzero().flatten().tolist()
    assert all(c < 10 for c in fully_zero_cols)
