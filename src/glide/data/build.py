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
from .audio import audio_num_samples
from .jsonl import JsonlDataset, load_hf_dataset
from .template import build_messages, build_prompt_messages

__all__ = ["build_sft_text_dataset", "build_rl_text_dataset", "build_multimodal_dataset"]


def build_sft_text_dataset(config: GlideConfig, split: str):
    """Build a conversational prompt/completion ``Dataset`` for text SFT."""
    path = getattr(config.data, split)
    if path is None:
        return None
    max_n = config.data.max_train_samples if split == "train" else config.data.max_eval_samples
    ds = load_hf_dataset(path, max_samples=max_n)

    data, template = config.data, config.template

    def _to_prompt_completion(rec: dict[str, Any]) -> dict[str, Any]:
        messages = build_messages(rec, data, template, Modality.TEXT)
        if messages and messages[-1].get("role") == "assistant":
            return {"prompt": messages[:-1], "completion": [messages[-1]]}
        # No assistant turn -> treat whole thing as a language-modeling sample.
        return {"prompt": messages, "completion": [{"role": "assistant", "content": ""}]}

    return ds.map(_to_prompt_completion, remove_columns=ds.column_names)


def build_rl_text_dataset(config: GlideConfig, split: str):
    """Build a ``{"prompt", ...extra columns...}`` ``Dataset`` for RL training."""
    path = getattr(config.data, split)
    if path is None:
        return None
    max_n = config.data.max_train_samples if split == "train" else config.data.max_eval_samples
    ds = load_hf_dataset(path, max_samples=max_n)

    data, template = config.data, config.template

    def _to_prompt(rec: dict[str, Any]) -> dict[str, Any]:
        return {"prompt": build_prompt_messages(rec, data, template, Modality.TEXT)}

    # Keep all original columns (reward funcs receive them as kwargs).
    return ds.map(_to_prompt)


def build_multimodal_dataset(config: GlideConfig, split: str):
    """Build a raw-record ``JsonlDataset`` for the speech/vision SFT path."""
    path = getattr(config.data, split)
    if path is None:
        return None
    max_n = config.data.max_train_samples if split == "train" else config.data.max_eval_samples
    return JsonlDataset(path, lazy=config.data.lazy, max_samples=max_n)


def _lengths_from_manifest(dataset: JsonlDataset, config: GlideConfig) -> list[int] | None:
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


def _manifest_paths(config: GlideConfig) -> list[str]:
    p = config.data.train
    if p is None:
        return []
    return [str(x) for x in (p if isinstance(p, list) else [p])]


def _lengths_cache_key(config: GlideConfig) -> str:
    """Fingerprint the manifests + relevant config so a stale cache is never reused."""
    parts: list[Any] = [config.data.audio_field, config.speech.sample_rate,
                        config.data.max_train_samples]
    for path in _manifest_paths(config):
        try:
            st = os.stat(path)
            parts.append((path, int(st.st_mtime), int(st.st_size)))
        except OSError:
            parts.append((path, None, None))
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _lengths_cache_path(config: GlideConfig) -> str | None:
    paths = _manifest_paths(config)
    if not paths:
        return None
    return paths[0] + ".glide-lengths.json"


def _read_lengths_cache(cache_path: str | None, key: str, n: int) -> list[int] | None:
    if not cache_path or not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path) as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return None
    if blob.get("key") == key and len(blob.get("lengths", [])) == n:
        return [int(x) for x in blob["lengths"]]
    return None


def _write_lengths_cache(cache_path: str | None, key: str, lengths: list[int]) -> None:
    if not cache_path:
        return
    tmp = f"{cache_path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump({"key": key, "lengths": lengths}, f)
        os.replace(tmp, cache_path)  # atomic
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _scan_audio_lengths(dataset: JsonlDataset, field: str, sr: int) -> list[int]:
    """Read one audio header per utterance (I/O-bound -> thread-pooled)."""
    from concurrent.futures import ThreadPoolExecutor

    refs = [dataset[i].get(field) for i in range(len(dataset))]

    def _one(ref):
        return audio_num_samples(ref, sr) if ref is not None else 0

    workers = min(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, refs))


def compute_audio_lengths(dataset: JsonlDataset, config: GlideConfig) -> list[int]:
    """Compute per-sample audio lengths (for the length-grouped batch sampler).

    Resolution order (fastest first): (1) manifest ``duration``/``num_samples`` field
    (zero I/O); (2) a disk cache keyed by manifest (path, mtime, size) + audio_field +
    sample_rate; (3) a thread-pooled header scan. Under DDP only rank 0 scans and writes
    the cache -- other ranks poll for it instead of each re-scanning the whole corpus.
    """
    field = config.data.audio_field
    sr = config.speech.sample_rate

    manifest = _lengths_from_manifest(dataset, config)
    if manifest is not None:
        return manifest

    n = len(dataset)
    key = _lengths_cache_key(config)
    cache_path = _lengths_cache_path(config)

    cached = _read_lengths_cache(cache_path, key, n)
    if cached is not None:
        return cached

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and rank != 0 and cache_path:
        # Let rank 0 do the (possibly minutes-long NFS) scan; poll for its cache rather
        # than launching N identical sweeps. Fall back to scanning locally on timeout.
        deadline = time.time() + 1800
        while time.time() < deadline:
            cached = _read_lengths_cache(cache_path, key, n)
            if cached is not None:
                return cached
            time.sleep(2.0)

    lengths = _scan_audio_lengths(dataset, field, sr)
    _write_lengths_cache(cache_path, key, lengths)
    return lengths
