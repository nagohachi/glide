"""Regression test for #10: lazy JSONL reopens handles on pid change (fork safety)."""

import json


def test_jsonl_reopens_handles_on_pid_change(tmp_path):
    from glide.data.jsonl import JsonlDataset

    path = tmp_path / "data.jsonl"
    with open(path, "w") as f:
        for r in [{"a": 1}, {"a": 2}, {"a": 3}]:
            f.write(json.dumps(r) + "\n")

    ds = JsonlDataset(str(path), lazy=True)
    f1 = ds._ensure_files()
    ds._files_pid = -1  # simulate having been forked into a new process
    f2 = ds._ensure_files()
    assert f1 is not f2  # reopened rather than sharing the parent's seek offset
    assert ds[1] == {"a": 2}
