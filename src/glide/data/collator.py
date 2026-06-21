"""Data collators with loss masking and (for text) packing support.

* :class:`MultimodalSFTCollator` -- builds a batch from raw JSONL records for the
  speech/vision SFT path. It applies the model's chat template, runs the HF
  processor to obtain ``input_ids`` plus modality features, and constructs
  ``labels`` with pad masking and optional *completion-only* masking. Completion
  masking is done by locating the tokenized ``response_template`` subsequence in
  the final ``input_ids`` -- the same model-agnostic trick TRL's
  ``DataCollatorForCompletionOnlyLM`` uses, which is robust to media-token
  expansion.

For the **text** path, ``glide`` defers to TRL's ``SFTTrainer`` collators
(packing, completion-only loss) configured through ``SFTConfig`` -- see
:mod:`glide.trainers.sft`.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any

import torch

from ..config.schema import DataConfig, Modality, TemplateConfig
from .audio import load_audio
from .template import build_messages

__all__ = ["MultimodalSFTCollator", "find_subsequence"]


def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Return the start index of the last occurrence of ``needle`` in ``haystack``.

    Returns ``-1`` when not found. Searching from the end matches the *final*
    assistant turn, which is the response we train on.
    """
    if not needle:
        return -1
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


def _detect_media_kwarg(processor, modality: Modality) -> str | None:
    """Pick the processor ``__call__`` kwarg name for the modality's media."""
    try:
        params = set(inspect.signature(processor.__call__).parameters)
    except (TypeError, ValueError):
        params = set()
    if modality is Modality.SPEECH:
        for name in ("audio", "audios", "raw_speech"):
            if name in params:
                return name
        return "audio"
    if modality is Modality.VISION:
        for name in ("images", "image"):
            if name in params:
                return name
        return "images"
    return None


@dataclass
class MultimodalSFTCollator:
    """Collate raw records into a padded, label-masked batch for multimodal SFT.

    Args:
        processor: An HF processor (tokenizer + feature extractor/image processor).
        data: Data field configuration.
        template: Template/masking configuration.
        modality: ``SPEECH`` or ``VISION``.
        sample_rate: Target audio sample rate.
        completion_only: Mask everything before the assistant response.
        response_template_ids: Pre-tokenized response template; if ``None`` it is
            derived from ``template.response_template``.
    """

    processor: Any
    data: DataConfig
    template: TemplateConfig
    modality: Modality
    sample_rate: int = 16000
    completion_only: bool = True
    response_template_ids: list[int] | None = None
    media_kwarg: str | None = field(default=None)

    def __post_init__(self):
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if self.media_kwarg is None:
            self.media_kwarg = _detect_media_kwarg(self.processor, self.modality)
        if (
            self.completion_only
            and self.response_template_ids is None
            and self.template.response_template
        ):
            self.response_template_ids = self.tokenizer.encode(
                self.template.response_template, add_special_tokens=False
            )

    def _load_media(self, record: dict):
        if self.modality is Modality.SPEECH and self.data.audio_field in record:
            return load_audio(record[self.data.audio_field], self.sample_rate)
        if self.modality is Modality.VISION and self.data.image_field in record:
            from PIL import Image

            ref = record[self.data.image_field]
            return Image.open(ref).convert("RGB") if isinstance(ref, str) else ref
        return None

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts, media = [], []
        for rec in records:
            messages = build_messages(rec, self.data, self.template, self.modality)
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
            m = self._load_media(rec)
            if m is not None:
                media.append(m)

        call_kwargs: dict[str, Any] = {
            "text": texts,
            "return_tensors": "pt",
            "padding": True,
        }
        if media and self.media_kwarg:
            call_kwargs[self.media_kwarg] = media
        if self.modality is Modality.SPEECH:
            call_kwargs.setdefault("sampling_rate", self.sample_rate)

        batch = self.processor(**call_kwargs)
        batch["labels"] = self._build_labels(batch)
        return batch

    def _build_labels(self, batch) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[input_ids == pad_id] = -100

        if self.completion_only and self.response_template_ids:
            rt = self.response_template_ids
            for row in range(input_ids.size(0)):
                ids = input_ids[row].tolist()
                start = find_subsequence(ids, rt)
                if start == -1:
                    labels[row, :] = -100  # response not found -> skip this row
                else:
                    labels[row, : start + len(rt)] = -100
        return labels
