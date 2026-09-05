"""Data collator for a composed Speech-LLM (encoder + projector + LLM).

Produces, per batch: text ``input_ids`` containing a single ``<audio>`` marker
(spliced with audio embeddings inside :class:`~glide.models.speech_llm.SpeechLLM`),
completion-only ``labels``, and the encoder's audio inputs
(``input_features``/``feature_attention_mask`` for Whisper/Qwen, or
``input_values``/``audio_attention_mask`` for WavLM/XLS-R). Applies speed
perturbation (waveform) and SpecAugment (log-mel features) at train time.
"""

import random
import warnings
from dataclasses import dataclass
from typing import Any

import torch

from ..config.schema import DataConfig, SpecAugmentConfig, SpeedPerturbConfig, TemplateConfig
from .audio import load_audio, speed_perturb
from .collator import find_subsequence
from .specaugment import spec_augment

__all__ = ["ComposedSpeechCollator"]


@dataclass
class ComposedSpeechCollator:
    tokenizer: Any
    feature_extractor: Any
    input_kind: str  # "input_features" | "input_values"
    audio_token: str
    data: DataConfig
    template: TemplateConfig
    sample_rate: int = 16000
    speed_perturb: SpeedPerturbConfig | None = None
    specaugment: SpecAugmentConfig | None = None
    completion_only: bool = True
    train: bool = True

    def __post_init__(self):
        self._resp_ids = None
        if self.completion_only and self.template.response_template:
            self._resp_ids = self.tokenizer.encode(
                self.template.response_template, add_special_tokens=False
            )

    def _speed(self, wav, rec):
        sp = self.speed_perturb
        if not (self.train and sp and sp.enabled):
            return wav
        factor = float(rec.get(sp.field_name, 1.0)) if sp.from_field else random.choice(sp.factors)
        return speed_perturb(wav, factor, self.sample_rate)

    def _ct_kwargs(self) -> dict:
        """apply_chat_template kwargs (only pass enable_thinking when configured)."""
        if self.template.enable_thinking is None:
            return {}
        return {"enable_thinking": self.template.enable_thinking}

    def _messages(self, rec, with_target: bool) -> list[dict]:
        prompt = rec.get(self.data.prompt_field, "")
        messages = []
        if self.template.system_prompt:
            messages.append({"role": "system", "content": self.template.system_prompt})
        messages.append({"role": "user", "content": self.audio_token + (prompt or "")})
        if with_target:
            target = rec.get(self.data.response_field)
            msg = {"role": "assistant", "content": target if target is not None else ""}
            # Thinking-mode SFT: pass reasoning via the chat template's reasoning_content
            # so the target renders as <think>{reasoning}</think>{answer}. Only when
            # thinking is enabled (else the model is trained to emit an empty think).
            reasoning = rec.get(self.data.reasoning_field)
            if reasoning and self.template.enable_thinking:
                msg["reasoning_content"] = reasoning
            messages.append(msg)
        return messages

    def _full_text(self, rec) -> str:
        return self.tokenizer.apply_chat_template(
            self._messages(rec, with_target=True), tokenize=False,
            add_generation_prompt=False, **self._ct_kwargs()
        )

    def _prompt_text(self, rec) -> str:
        """Prompt up to (and including) the assistant header -- for prefix masking.

        Uses the SAME enable_thinking as the full render so the prefix stays a true
        prefix of the target (critical: with enable_thinking=False the prompt already
        contains the empty ``<think></think>``, which must match the target).
        """
        return self.tokenizer.apply_chat_template(
            self._messages(rec, with_target=False), tokenize=False,
            add_generation_prompt=True, **self._ct_kwargs()
        )

    def generation_inputs(self, records: list[dict]) -> dict:
        """Prompt-only inputs for AR validation generation (no target/labels).

        Uses the SAME prompt + audio-feature construction as training so eval matches
        training. No speed-perturb/specaugment (eval). Intended for batch size 1
        (``SpeechLLM._splice`` right-pads, which breaks batched generation).
        """
        prompt_texts = [self._prompt_text(r) for r in records]
        wavs = [load_audio(r[self.data.audio_field], self.sample_rate) for r in records]
        enc = self.tokenizer(prompt_texts, return_tensors="pt", padding=True,
                             add_special_tokens=False)
        batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        feats = self.feature_extractor(wavs, sampling_rate=self.sample_rate, return_tensors="pt")
        if self.input_kind == "input_features":
            batch["input_features"] = feats["input_features"]
            if "attention_mask" in feats:
                batch["feature_attention_mask"] = feats["attention_mask"]
        else:
            batch["input_values"] = feats["input_values"]
            if "attention_mask" in feats:
                batch["audio_attention_mask"] = feats["attention_mask"]
        return batch

    def __call__(self, records: list[dict]) -> dict:
        texts = [self._full_text(r) for r in records]
        wavs = [self._speed(load_audio(r[self.data.audio_field], self.sample_rate), r)
                for r in records]
        # Per-sample prompt token length, for model-agnostic completion-only masking
        # (no response_template marker needed -- the chat template defines the boundary).
        prefix_lens = [
            len(self.tokenizer(self._prompt_text(r), add_special_tokens=False)["input_ids"])
            for r in records
        ]

        # add_special_tokens=False: the chat template already renders every special
        # token (incl. BOS for Llama/Gemma). Letting the tokenizer add another BOS
        # here would (a) double-BOS the sequence, (b) shift it one token past the
        # add_special_tokens=False prefix_lens (masking the last prompt token as a
        # target), and (c) diverge from generation_inputs (also add_special_tokens=False).
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        batch["labels"] = self._labels(enc["input_ids"], prefix_lens)

        feats = self.feature_extractor(wavs, sampling_rate=self.sample_rate, return_tensors="pt")
        if self.input_kind == "input_features":
            features = feats["input_features"]
            if self.train and self.specaugment is not None and self.specaugment.enabled:
                features = spec_augment(
                    features, self.specaugment, attention_mask=feats.get("attention_mask")
                )
            batch["input_features"] = features
            if "attention_mask" in feats:
                batch["feature_attention_mask"] = feats["attention_mask"]
        else:
            if self.train and self.specaugment is not None and self.specaugment.enabled:
                warnings.warn(
                    "speech.specaugment is enabled but the encoder consumes raw waveforms "
                    "(input_values, e.g. WavLM/XLS-R); SpecAugment only applies to log-mel "
                    "input_features and is being ignored.",
                    stacklevel=2,
                )
            batch["input_values"] = feats["input_values"]
            if "attention_mask" in feats:
                batch["audio_attention_mask"] = feats["attention_mask"]
        return batch

    def _labels(self, input_ids, prefix_lens) -> torch.Tensor:
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[input_ids == pad_id] = -100
        if not self.completion_only:
            return labels
        if self._resp_ids:  # explicit marker (back-compat): mask up to its last occurrence
            for row in range(input_ids.shape[0]):
                start = find_subsequence(input_ids[row].tolist(), self._resp_ids)
                labels[row, :] = -100 if start == -1 else labels[row, :]
                if start != -1:
                    labels[row, : start + len(self._resp_ids)] = -100
        else:  # default: mask the chat-template prompt prefix (marker-free, model-agnostic)
            for row, plen in enumerate(prefix_lens):
                labels[row, :plen] = -100
        return labels
