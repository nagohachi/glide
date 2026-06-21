"""Tests for sequence-packing geometry (mask / cu_seqlens / position_ids).

These cover the tricky, correctness-critical logic shared by the packing collators
(the plugin glue itself is validated on GPU by the loss-parity tests).
"""

import torch

from glide.data.packing import (
    block_diagonal_causal_mask,
    cu_seqlens,
    packed_position_ids,
    segment_ids,
)


def test_packed_position_ids_reset_per_example():
    assert packed_position_ids([3, 2]).tolist() == [0, 1, 2, 0, 1]


def test_segment_ids():
    assert segment_ids([2, 3]).tolist() == [0, 0, 1, 1, 1]


def test_cu_seqlens_offsets_and_maxlen():
    cu, max_len = cu_seqlens([3, 2, 4])
    assert cu.tolist() == [0, 3, 5, 9]
    assert cu.dtype == torch.int32
    assert max_len == 4


def test_block_diagonal_mask_shape_and_blocking():
    m = block_diagonal_causal_mask([2, 2])[0, 0]  # (4,4) bool
    assert m.shape == (4, 4)
    # within example 0: causal
    assert m[0, 0] and not m[0, 1] and m[1, 0] and m[1, 1]
    # example 1 (rows/cols 2,3) must NOT attend to example 0 (cols 0,1)
    assert not m[2, 0] and not m[2, 1] and not m[3, 0]
    # within example 1: causal
    assert m[2, 2] and not m[2, 3] and m[3, 2] and m[3, 3]


def test_mask_and_cu_seqlens_agree():
    """The dense mask and the cu_seqlens segmentation must describe the same blocks."""
    seg_lengths = [3, 1, 4]
    mask = block_diagonal_causal_mask(seg_lengths)[0, 0]
    cu, _ = cu_seqlens(seg_lengths)
    total = int(cu[-1])
    seg = segment_ids(seg_lengths)
    for i in range(total):
        for j in range(total):
            expected = (j <= i) and (seg[i] == seg[j])
            assert bool(mask[i, j]) == expected
