"""JSONL dataset loading.

Two flavours are provided:

* :class:`JsonlDataset` -- a memory-light, map-style ``torch.utils.data.Dataset``
  that indexes records by byte offset and parses them on access (``lazy=True``),
  or loads everything up-front (``lazy=False``). Used for the multimodal path
  where a custom collator consumes raw records.
* :func:`load_hf_dataset` -- wraps one or more JSONL files in a 🤗 ``datasets``
  ``Dataset``, which is what the TRL text trainers expect.

Records are arbitrary JSON objects; field names are interpreted by the active
:class:`~glide.config.schema.DataConfig`.
"""

import json
from pathlib import Path
from typing import Any, Sequence

from torch.utils.data import Dataset

__all__ = ["JsonlDataset", "load_hf_dataset", "read_jsonl"]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Eagerly read a JSONL file into a list of dicts (blank lines skipped)."""
    rows: list[dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class JsonlDataset(Dataset[dict[str, Any]]):
    """A map-style dataset over one or more JSONL files.

    Args:
        paths: One path or a list of paths. Files are concatenated.
        lazy: If ``True`` (default), only byte offsets are held in memory and each
            record is parsed on ``__getitem__``. If ``False`` everything is parsed
            up-front.
        max_samples: Optionally cap the number of records (after concatenation).
        transform: Optional callable applied to each parsed record.
    """

    def __init__(
        self,
        paths: str | Sequence[str],
        *,
        lazy: bool = True,
        max_samples: int | None = None,
        transform=None,
    ):
        if isinstance(paths, (str, Path)):
            paths = [str(paths)]
        self.paths = [str(p) for p in paths]
        self.lazy = lazy
        self.transform = transform

        # (file_index, byte_offset) per record, or parsed records when not lazy.
        self._index: list[tuple[int, int]] = []
        self._records: list[dict[str, Any]] | None = None
        self._files = None

        if lazy:
            for fi, p in enumerate(self.paths):
                with open(p, "rb") as f:
                    offset = f.tell()
                    line = f.readline()
                    while line:
                        if line.strip():
                            self._index.append((fi, offset))
                        offset = f.tell()
                        line = f.readline()
            if max_samples is not None:
                self._index = self._index[:max_samples]
        else:
            recs: list[dict[str, Any]] = []
            for p in self.paths:
                recs.extend(read_jsonl(p))
            if max_samples is not None:
                recs = recs[:max_samples]
            self._records = recs

    def __len__(self) -> int:
        return len(self._index) if self.lazy else len(self._records or [])

    def _ensure_files(self):
        # Open files lazily and per-process (safe across dataloader workers).
        if self._files is None:
            self._files = [open(p, "rb") for p in self.paths]
        return self._files

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.lazy:
            fi, offset = self._index[index]
            f = self._ensure_files()[fi]
            f.seek(offset)
            record = json.loads(f.readline().decode("utf-8"))
        else:
            assert self._records is not None  # eager mode populates this
            record = self._records[index]
        return self.transform(record) if self.transform else record

    def __getstate__(self):
        # Drop open file handles so workers can pickle the dataset.
        state = self.__dict__.copy()
        state["_files"] = None
        return state


def load_hf_dataset(paths: str | Sequence[str], *, max_samples: int | None = None):
    """Load JSONL file(s) into a 🤗 ``datasets.Dataset`` (for TRL text trainers)."""
    from datasets import load_dataset

    if isinstance(paths, (str, Path)):
        paths = [str(paths)]
    ds = load_dataset("json", data_files=[str(p) for p in paths], split="train")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds
