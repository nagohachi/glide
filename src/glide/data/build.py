"""Assemble datasets (and, for multimodal, the collator) from a config.

The shape of the data depends on the task/modality:

* **Text SFT** -> a 🤗 ``Dataset`` of ``{"prompt", "completion"}`` *conversational*
  rows. TRL's ``SFTTrainer`` applies the chat template and masks the prompt
  (completion-only loss) itself.
* **Text RL** (GRPO/GSPO) -> a 🤗 ``Dataset`` of ``{"prompt", ...}`` rows; all
  other JSONL columns are preserved and handed to reward functions.
* **Multimodal SFT** (speech/vision) -> a :class:`~glide.data.jsonl.JsonlDataset`
  of raw records consumed by :class:`~glide.data.collator.MultimodalSFTCollator`.
"""

import hashlib
import json
import os
import time
from typing import Any

from ..config.schema import GlideConfig, Modality
from .audio import count_audio_samples
from .jsonl import JsonlDataset, load_hf_dataset
from .template import build_messages, build_prompt_messages

__all__ = ["build_sft_text_dataset", "build_rl_text_dataset", "build_multimodal_dataset"]


def build_sft_text_dataset(config: GlideConfig, split: str):
    """Build a conversational prompt/completion ``Dataset`` for text SFT."""
    path, max_n = _get_split_source(config, split)
    if path is None:
        return None
    ds = load_hf_dataset(path, max_samples=max_n)

    data, template = config.data, config.template

    def _convert_to_prompt_completion(rec: dict[str, Any]) -> dict[str, Any]:
        messages = build_messages(rec, data, template, Modality.TEXT)
        if messages and messages[-1].get("role") == "assistant":
            return {"prompt": messages[:-1], "completion": [messages[-1]]}
        # No assistant turn -> treat whole thing as a language-modeling sample.
        return {"prompt": messages, "completion": [{"role": "assistant", "content": ""}]}

    return ds.map(_convert_to_prompt_completion, remove_columns=ds.column_names)


def build_rl_text_dataset(config: GlideConfig, split: str):
    """Build a ``{"prompt", ...extra columns...}`` ``Dataset`` for RL training."""
    path, max_n = _get_split_source(config, split)
    if path is None:
        return None
    ds = load_hf_dataset(path, max_samples=max_n)

    data, template = config.data, config.template

    def _convert_to_prompt(rec: dict[str, Any]) -> dict[str, Any]:
        return {"prompt": build_prompt_messages(rec, data, template, Modality.TEXT)}

    # Keep all original columns (reward funcs receive them as kwargs).
    return ds.map(_convert_to_prompt)


def build_multimodal_dataset(config: GlideConfig, split: str):
    """Build a raw-record ``JsonlDataset`` for the speech/vision SFT path."""
    path, max_n = _get_split_source(config, split)
    if path is None:
        return None
    return JsonlDataset(path, lazy=config.data.lazy, max_samples=max_n)


def _get_split_source(
    config: GlideConfig, split: str
) -> tuple[str | list[str] | None, int | None]:
    """Return ``(data.{split}_jsonl_path, sample cap)`` for ``split`` (``train``/``eval``)."""
    path = getattr(config.data, f"{split}_jsonl_path")
    max_n = config.data.max_train_samples if split == "train" else config.data.max_eval_samples
    return path, max_n


def _read_lengths_from_manifest(dataset: JsonlDataset, config: GlideConfig) -> list[int] | None:
    """Return per-sample lengths from a manifest duration/num_samples field, or ``None``.

    Zero file I/O -- most ASR manifests already carry ``duration``. Returns ``None`` if
    the configured field is absent from any record (so we fall back to a header scan).
    """
    sr = config.speech.sample_rate
    dur_field = config.data.duration_field
    nsamp_field = config.data.num_samples_field
    if not (dur_field or nsamp_field):
        return None
    lengths: list[int] = []
    for i in range(len(dataset)):
        rec = dataset[i]
        if nsamp_field and nsamp_field in rec and rec[nsamp_field] is not None:
            lengths.append(int(rec[nsamp_field]))
        elif dur_field and dur_field in rec and rec[dur_field] is not None:
            lengths.append(int(float(rec[dur_field]) * sr))
        else:
            return None  # incomplete -> fall back to scanning
    return lengths


def _list_train_manifest_paths(config: GlideConfig) -> list[str]:
    """Return ``data.train_jsonl_path`` as a list of strings (empty when unset)."""
    p = config.data.train_jsonl_path
    if p is None:
        return []
    return [str(x) for x in (p if isinstance(p, list) else [p])]


class _AudioLengthCache:
    """On-disk cache of per-sample audio lengths, stored next to the first manifest.

    ``key`` fingerprints the manifests (path, mtime, size) and the config fields that
    change the lengths, so a stale scan is never reused. ``path`` is ``None`` when no
    train manifest is configured; every read then misses and writes are skipped.
    """

    def __init__(self, path: str | None, key: str):
        self.path = path
        self.key = key

    @classmethod
    def build_from_config(cls, config: GlideConfig) -> "_AudioLengthCache":
        manifests = _list_train_manifest_paths(config)
        parts: list[Any] = [config.data.audio_field, config.speech.sample_rate,
                            config.data.max_train_samples]
        for manifest in manifests:
            try:
                st = os.stat(manifest)
                parts.append((manifest, int(st.st_mtime), int(st.st_size)))
            except OSError:
                parts.append((manifest, None, None))
        key = hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()
        path = manifests[0] + ".glide-lengths.json" if manifests else None
        return cls(path, key)

    def read(self, num_samples: int) -> list[int] | None:
        """Return the cached lengths, or ``None`` on a missing/stale/corrupt cache."""
        if not self.path or not os.path.isfile(self.path):
            return None
        try:
            with open(self.path) as f:
                blob = json.load(f)
        except (OSError, ValueError):
            return None
        if blob.get("key") == self.key and len(blob.get("lengths", [])) == num_samples:
            return [int(x) for x in blob["lengths"]]
        return None

    def write(self, lengths: list[int]) -> None:
        """Atomically replace the cache file (best effort: I/O errors are swallowed)."""
        if not self.path:
            return
        tmp = f"{self.path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as f:
                json.dump({"key": self.key, "lengths": lengths}, f)
            os.replace(tmp, self.path)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _scan_audio_lengths(dataset: JsonlDataset, field: str, sr: int) -> list[int]:
    """Read one audio header per utterance (I/O-bound -> thread-pooled)."""
    from concurrent.futures import ThreadPoolExecutor

    refs = [dataset[i].get(field) for i in range(len(dataset))]

    def _count_samples(ref):
        return count_audio_samples(ref, sr) if ref is not None else 0

    workers = min(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_count_samples, refs))


def compute_audio_lengths(dataset: JsonlDataset, config: GlideConfig) -> list[int]:
    """Compute per-sample audio lengths (for the length-grouped batch sampler).

    Resolution order (fastest first): (1) manifest ``duration``/``num_samples`` field
    (zero I/O); (2) a disk cache keyed by manifest (path, mtime, size) + audio_field +
    sample_rate; (3) a thread-pooled header scan. Under DDP only rank 0 scans and writes
    the cache -- other ranks poll for it instead of each re-scanning the whole corpus.
    """
    field = config.data.audio_field
    sr = config.speech.sample_rate

    manifest = _read_lengths_from_manifest(dataset, config)
    if manifest is not None:
        return manifest

    n = len(dataset)
    cache = _AudioLengthCache.build_from_config(config)
    cached = cache.read(n)
    if cached is not None:
        return cached

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and rank != 0 and cache.path:
        # Let rank 0 do the (possibly minutes-long NFS) scan; poll for its cache rather
        # than launching N identical sweeps. Fall back to scanning locally on timeout.
        deadline = time.time() + 1800
        while time.time() < deadline:
            cached = cache.read(n)
            if cached is not None:
                return cached
            time.sleep(2.0)

    lengths = _scan_audio_lengths(dataset, field, sr)
    cache.write(lengths)
    return lengths
