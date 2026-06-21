"""Tests for JSONL dataset loading (lazy byte-offset vs eager)."""

import pickle

import pytest

from glide.data import JsonlDataset, read_jsonl


RECORDS = [{"id": i, "text": f"line {i}"} for i in range(10)]


@pytest.mark.parametrize("lazy", [True, False])
def test_roundtrip(jsonl_file, lazy):
    path = jsonl_file(RECORDS)
    ds = JsonlDataset(path, lazy=lazy)
    assert len(ds) == 10
    assert ds[0] == {"id": 0, "text": "line 0"}
    assert ds[9] == {"id": 9, "text": "line 9"}


def test_max_samples(jsonl_file):
    path = jsonl_file(RECORDS)
    assert len(JsonlDataset(path, lazy=True, max_samples=3)) == 3
    assert len(JsonlDataset(path, lazy=False, max_samples=3)) == 3


def test_multiple_files_concatenated(jsonl_file):
    p1 = jsonl_file(RECORDS[:5], "a.jsonl")
    p2 = jsonl_file(RECORDS[5:], "b.jsonl")
    ds = JsonlDataset([p1, p2], lazy=True)
    assert len(ds) == 10
    assert ds[7]["id"] == 7


def test_blank_lines_skipped(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert len(JsonlDataset(str(path), lazy=True)) == 2
    assert len(read_jsonl(str(path))) == 2


def test_lazy_dataset_is_picklable(jsonl_file):
    # Workers pickle the dataset; open file handles must be dropped.
    ds = JsonlDataset(jsonl_file(RECORDS), lazy=True)
    _ = ds[0]
    restored = pickle.loads(pickle.dumps(ds))
    assert restored[1]["id"] == 1


def test_transform_applied(jsonl_file):
    ds = JsonlDataset(jsonl_file(RECORDS), lazy=False, transform=lambda r: r["id"])
    assert ds[3] == 3
