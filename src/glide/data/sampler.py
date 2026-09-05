"""Length-grouped batch sampler for the speech modality.

Speech batches are dominated by padding when samples of very different durations
are mixed. The spec calls for: *sort samples by length, form batches of
similar-length samples, but vary the order across epochs*.

:class:`LengthGroupedBatchSampler` implements the standard "mega-batch" trick:

1. Shuffle all indices with the epoch seed.
2. Cut them into mega-batches of ``batch_size * mega_batch_mult``.
3. Sort each mega-batch by length (so a batch is locally homogeneous).
4. Cut each mega-batch into ``batch_size`` batches.
5. Shuffle the order of the resulting batches.

Because step 1 and step 5 are reseeded every epoch, both the *composition* of a
batch and the *order* of batches change from epoch to epoch, while padding stays
low. Call :meth:`set_epoch` once per epoch (the trainer does this automatically).
"""

import random
from typing import Iterator, Sequence

from torch.utils.data import Sampler

__all__ = ["LengthGroupedBatchSampler"]


class LengthGroupedBatchSampler(Sampler[list[int]]):
    """Yield batches (lists of indices) of similar length, reshuffled per epoch.

    Two batching modes:

    * **Fixed** -- exactly ``batch_size`` samples per batch.
    * **Dynamic** (``max_tokens`` set) -- as many samples as fit within a
      ``max_tokens`` budget (sum of lengths), so batches of short utterances hold
      more samples than batches of long ones. This is the common speech setup and
      means the per-device batch size *varies*. ``batch_size`` then acts as an
      optional hard cap on the count.

    Args:
        lengths: Per-sample lengths (e.g. number of audio frames or tokens).
        batch_size: Samples per batch (fixed mode), or a count cap (dynamic mode).
        max_tokens: If set, enables dynamic batching with this length budget.
        mega_batch_mult: Mega-batch size as a multiple of ``batch_size``. Larger
            values group more aggressively (less padding) but reduce randomness.
        drop_last: Drop a trailing partial batch (fixed mode only).
        seed: Base RNG seed; combined with the epoch for per-epoch shuffling.
        shuffle: When ``False``, samples are sorted globally by length with no
            per-epoch shuffling (useful for deterministic evaluation).
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        max_tokens: int | None = None,
        mega_batch_mult: int = 50,
        drop_last: bool = False,
        seed: int = 0,
        shuffle: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if num_replicas < 1 or not (0 <= rank < num_replicas):
            raise ValueError("Invalid num_replicas/rank for distributed sampling.")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.mega_batch_mult = max(1, mega_batch_mult)
        self.drop_last = drop_last
        self.seed = seed
        self.shuffle = shuffle
        #: Distributed sharding: under DDP each of ``num_replicas`` ranks consumes
        #: a disjoint stripe of batches (``batches[rank::num_replicas]``), so no
        #: sample is seen twice per epoch and every rank gets the same count.
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self._cached_len: int | None = None

    def _batches(self) -> list[list[int]]:
        n = len(self.lengths)
        indices = list(range(n))

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)

        # shuffle=False -> a single mega-batch spanning the whole dataset, so the sort
        # below is a *global* length sort (deterministic, minimal padding) as documented,
        # not merely per-megabatch.
        mega = n if not self.shuffle else self.batch_size * self.mega_batch_mult
        megabatches = [indices[i : i + mega] for i in range(0, n, mega)]
        # Sort each mega-batch by descending length -> homogeneous local batches.
        megabatches = [
            sorted(mb, key=lambda i: self.lengths[i], reverse=True) for mb in megabatches
        ]

        batches: list[list[int]] = []
        for mb in megabatches:
            if self.max_tokens is not None:
                batches.extend(self._dynamic_batches(mb))
            else:
                for i in range(0, len(mb), self.batch_size):
                    batch = mb[i : i + self.batch_size]
                    if self.drop_last and len(batch) < self.batch_size:
                        continue
                    batches.append(batch)

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch + 1)
            rng.shuffle(batches)

        if self.num_replicas > 1:
            # Truncate to a multiple of num_replicas so every rank gets the same
            # number of batches, then take this rank's disjoint stripe.
            even = len(batches) - (len(batches) % self.num_replicas)
            batches = batches[self.rank : even : self.num_replicas]
        return batches

    def _dynamic_batches(self, mb: list[int]) -> list[list[int]]:
        """Pack a (length-sorted) mega-batch into token-budgeted batches."""
        assert self.max_tokens is not None  # only called in dynamic mode
        budget = self.max_tokens
        out: list[list[int]] = []
        cur: list[int] = []
        for idx in mb:
            ln = self.lengths[idx]
            over_budget = cur and sum(self.lengths[i] for i in cur) + ln > budget
            over_count = len(cur) >= self.batch_size
            if cur and (over_budget or over_count):
                out.append(cur)
                cur = []
            cur.append(idx)
        if cur:
            out.append(cur)
        return out

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        # Dynamic budgeting and distributed striping make the count depend on the
        # (epoch-dependent) grouping, so compute it once and cache.
        if self.max_tokens is not None or self.num_replicas > 1:
            if self._cached_len is None:
                self._cached_len = len(self._batches())
            return self._cached_len
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so successive epochs produce different orderings."""
        self.epoch = epoch
        self._cached_len = None
