"""Geometry helpers for sequence packing (model-agnostic, pure tensor ops).

When several examples are concatenated into one unpadded row, the model must (a)
know where each example starts/ends and (b) not attend across those boundaries.
These helpers produce the three artifacts that encode that, given the per-example
token lengths ``seg_lengths``:

* :func:`packed_position_ids` -- positions that reset to 0 at each boundary.
* :func:`cu_seqlens` -- cumulative segment offsets for the FlashAttention-2
  *varlen* kernel (the efficient, O(sum Lᵢ²) path).
* :func:`block_diagonal_causal_mask` -- a dense boolean mask for SDPA/eager
  (correct but O(T²) over the whole pack).

The two attention representations are equivalent (see
:func:`block_diagonal_causal_mask` vs :func:`cu_seqlens`); which one to use
depends on the attention backend.
"""

from typing import Sequence

import torch

__all__ = ["packed_position_ids", "cu_seqlens", "block_diagonal_causal_mask", "segment_ids"]


def segment_ids(seg_lengths: Sequence[int]) -> torch.Tensor:
    """Return a 1-D tensor mapping each packed position to its example index."""
    return torch.cat([torch.full((int(n),), i, dtype=torch.long)
                      for i, n in enumerate(seg_lengths)])


def packed_position_ids(seg_lengths: Sequence[int]) -> torch.Tensor:
    """Return position ids ``[0..L0-1, 0..L1-1, ...]`` (shape ``(T,)``)."""
    return torch.cat([torch.arange(int(n)) for n in seg_lengths])


def cu_seqlens(seg_lengths: Sequence[int]) -> tuple[torch.Tensor, int]:
    """Return ``(cu, max_len)`` for FA2 varlen: ``cu = [0, L0, L0+L1, ...]`` int32."""
    cu = torch.zeros(len(seg_lengths) + 1, dtype=torch.int32)
    acc = 0
    for i, n in enumerate(seg_lengths):
        acc += int(n)
        cu[i + 1] = acc
    return cu, (int(max(seg_lengths)) if len(seg_lengths) else 0)


def block_diagonal_causal_mask(seg_lengths: Sequence[int]) -> torch.Tensor:
    """Return a ``(1, 1, T, T)`` boolean mask (True = attend).

    Position ``i`` may attend to ``j`` iff ``j <= i`` (causal) AND ``i`` and ``j``
    are in the same example (block-diagonal).
    """
    total = int(sum(seg_lengths))
    seg = segment_ids(seg_lengths)
    causal = torch.tril(torch.ones(total, total, dtype=torch.bool))
    same = seg[:, None] == seg[None, :]
    return (causal & same)[None, None]
