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


def compute_audio_lengths(dataset: JsonlDataset, config: GlideConfig) -> list[int]:
    """Compute per-sample audio lengths (for the length-grouped batch sampler)."""
    field = config.data.audio_field
    sr = config.speech.sample_rate
    lengths = []
    for i in range(len(dataset)):
        rec = dataset[i]
        ref = rec.get(field)
        lengths.append(audio_num_samples(ref, sr) if ref is not None else 0)
    return lengths
