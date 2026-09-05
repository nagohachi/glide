"""Tests for the speech length-grouped batch sampler."""

from glide.data import LengthGroupedBatchSampler


def test_covers_all_indices_exactly_once():
    lengths = list(range(100))
    s = LengthGroupedBatchSampler(lengths, batch_size=8, seed=0)
    seen = [i for batch in s for i in batch]
    assert sorted(seen) == list(range(100))


def test_len_matches_iteration():
    lengths = list(range(95))
    s = LengthGroupedBatchSampler(lengths, batch_size=8)
    assert len(list(s)) == len(s)


def test_batches_are_length_homogeneous():
    # With small mega-batches, each batch should hold similar lengths.
    lengths = [i for i in range(64)]
    s = LengthGroupedBatchSampler(lengths, batch_size=8, mega_batch_mult=1, seed=1, shuffle=False)
    for batch in s:
        spread = max(lengths[i] for i in batch) - min(lengths[i] for i in batch)
        assert spread <= 8  # within one batch worth of contiguous lengths


def test_order_varies_across_epochs_but_partition_stable_size():
    lengths = list(range(100))
    s = LengthGroupedBatchSampler(lengths, batch_size=8, seed=0)
    s.set_epoch(0)
    e0 = list(s)
    s.set_epoch(1)
    e1 = list(s)
    # Different epochs -> different ordering / composition.
    assert e0 != e1
    # But both still cover everything.
    assert sorted(i for b in e1 for i in b) == list(range(100))


def test_drop_last():
    lengths = list(range(10))
    s = LengthGroupedBatchSampler(lengths, batch_size=4, drop_last=True)
    batches = list(s)
    assert all(len(b) == 4 for b in batches)
    assert len(batches) == 2  # 10 // 4


def test_dynamic_batching_respects_token_budget():
    # Uniform length 10, budget 100 -> at most 10 samples per batch.
    lengths = [10] * 95
    s = LengthGroupedBatchSampler(lengths, batch_size=1000, max_tokens=100, seed=0)
    batches = list(s)
    assert sorted(i for b in batches for i in b) == list(range(95))
    for b in batches:
        assert sum(lengths[i] for i in b) <= 100


def test_dynamic_batching_varies_count_by_length():
    # Short utterances should pack more per batch than long ones.
    lengths = [1] * 50 + [50] * 50
    s = LengthGroupedBatchSampler(lengths, batch_size=1000, max_tokens=50,
                                  mega_batch_mult=1, seed=0, shuffle=False)
    sizes = [len(b) for b in s]
    assert max(sizes) > min(sizes)  # dynamic batch sizes
    assert len(s) == len(list(s))   # cached len matches


def test_distributed_stripes_are_disjoint_and_equal():
    lengths = list(range(200))
    samplers = [
        LengthGroupedBatchSampler(lengths, batch_size=8, seed=0, num_replicas=4, rank=r)
        for r in range(4)
    ]
    per_rank = [list(s) for s in samplers]
    # Equal batch count per rank (DDP requires identical step counts).
    assert len({len(b) for b in per_rank}) == 1
    # No batch (by identity of its sample set) appears on two ranks.
    seen_batches = [tuple(sorted(b)) for batches in per_rank for b in batches]
    assert len(seen_batches) == len(set(seen_batches))


def test_deterministic_given_seed_and_epoch():
    lengths = list(range(50))
    a = LengthGroupedBatchSampler(lengths, batch_size=5, seed=7)
    b = LengthGroupedBatchSampler(lengths, batch_size=5, seed=7)
    a.set_epoch(2)
    b.set_epoch(2)
    assert list(a) == list(b)
