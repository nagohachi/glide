"""Validation-time autoregressive decoding and metric computation.

The :class:`GenerationEvaluator` runs ``model.generate`` over the eval records,
decodes the *newly generated* tokens (stripping the prompt), and scores them with
the configured metrics (WER/CER/BLEU/ROUGE). It works for text, speech and vision
models because it builds generation inputs through the same chat template /
processor used at training time, asking for a generation prompt and omitting the
response.

:class:`GenerateEvalCallback` plugs the evaluator into a 🤗 ``Trainer`` so metrics
appear in the training logs / wandb / tensorboard each evaluation.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from transformers import TrainerCallback

from ..config.schema import GlideConfig, Modality
from ..data.collator import load_media
from ..data.template import build_prompt_messages, extract_reference
from ..logging_utils import get_logger
from ..metrics.text_metrics import build_metric_fn

_log = get_logger("eval")

__all__ = ["GenerationEvaluator", "GenerateEvalCallback", "TestEvalCallback"]


class GenerationEvaluator:
    """Generate from eval records and compute text metrics."""

    def __init__(
        self,
        config: GlideConfig,
        processor,
        records: Sequence[dict],
        cap_samples: bool = True,
    ):
        self.config = config
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.records = list(records)
        self.metric_fn = build_metric_fn(
            config.eval.metrics, normalize=config.eval.normalize_text
        )
        gen = config.eval.generate
        # ``max_eval_samples`` caps validation-time decoding (generation is slow);
        # held-out test evaluation passes cap_samples=False to score the full set.
        max_n = config.data.max_eval_samples
        if cap_samples and max_n is not None:
            self.records = self.records[:max_n]
        self.gen_kwargs = dict(
            max_new_tokens=gen.max_new_tokens,
            num_beams=gen.num_beams,
            do_sample=gen.do_sample,
            temperature=gen.temperature,
            top_p=gen.top_p,
        )
        self.batch_size = gen.batch_size
        # Composed Speech-LLM has no multimodal processor -> build inputs via the collator,
        # and generate at batch size 1 (SpeechLLM._splice_audio_embeddings right-pads, which
        # breaks batched generation).
        self._composed = config.is_composed_speech_llm()
        self._composed_collator = None
        if self._composed:
            self.batch_size = 1
        self.answer_after = config.eval.answer_after
        self.answer_regex = (
            re.compile(config.eval.answer_regex, re.DOTALL)
            if config.eval.answer_regex
            else None
        )

    def _extract_answer(self, text: str) -> str:
        """Select the scored substring per ``answer_after`` / ``answer_regex``.

        ``answer_after`` keeps the text after the *last* occurrence of the literal
        delimiter; ``answer_regex`` then keeps group 1 (or the whole match if the
        pattern has no groups). Either step is a no-op if its pattern is absent or
        does not match, so non-think output passes through unchanged.
        """
        if self.answer_after is not None:
            idx = text.rfind(self.answer_after)
            if idx != -1:
                text = text[idx + len(self.answer_after):]
        if self.answer_regex is not None:
            m = self.answer_regex.search(text)
            if m is not None:
                text = m.group(1) if m.groups() else m.group(0)
        return text.strip()

    def _build_inputs(self, batch_records: list[dict], device):
        """Build left-padded generation inputs for a batch of records."""
        if self._composed:
            assert self._composed_collator is not None
            inputs = self._composed_collator.build_generation_inputs(batch_records)
            return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        modality = self.config.modality
        # Only pass enable_thinking when configured (matches ComposedSpeechCollator).
        chat_template_kwargs: dict = {}
        if self.config.template.enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = self.config.template.enable_thinking
        texts, media = [], []
        for rec in batch_records:
            msgs = build_prompt_messages(rec, self.config.data, self.config.template, modality)
            texts.append(
                self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
                )
            )
            item = load_media(rec, self.config.data, modality, self.config.speech.sample_rate)
            if item is not None:
                media.append(item)

        # Left padding is required for correct batched generation. Restore in a finally
        # so an exception mid-eval doesn't leave the shared tokenizer in left-padding
        # mode for the training collators.
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        call_kwargs: dict[str, Any] = {"text": texts, "return_tensors": "pt", "padding": True}
        if media:
            kw = "audio" if modality is Modality.SPEECH else "images"
            call_kwargs[kw] = media
            if modality is Modality.SPEECH:
                call_kwargs["sampling_rate"] = self.config.speech.sample_rate
        try:
            inputs = self.processor(**call_kwargs)
        finally:
            self.tokenizer.padding_side = prev_side
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    @staticmethod
    def _extract_references(records, data_cfg) -> list[str]:
        return [extract_reference(r, data_cfg) or "" for r in records]

    def evaluate(
        self,
        model,
        save_path: str | Path | None = None,
        step: int | None = None,
        prefix: str = "eval",
    ) -> dict[str, float]:
        """Run generation over all eval records and return the metric dict.

        If *save_path* is given the per-sample predictions are written there as
        JSONL (overwriting any previous file so only the latest run is kept).
        Metric keys are prefixed with *prefix* (``"eval"`` during training,
        ``"test"`` at the end-of-training test evaluation).
        """
        import torch

        gen_model = getattr(model, "module", model)  # unwrap DDP for .generate/.encoder
        if self._composed and self._composed_collator is None:
            from ..data.composed_collator import ComposedSpeechCollator

            self._composed_collator = ComposedSpeechCollator.build_from_config(
                self.config, self.tokenizer, gen_model.encoder, train=False
            )

        model_was_training = gen_model.training
        gen_model.eval()
        device = next(gen_model.parameters()).device
        predictions: list[str] = []

        with torch.no_grad():
            for i in range(0, len(self.records), self.batch_size):
                batch = self.records[i : i + self.batch_size]
                inputs = self._build_inputs(batch, device)
                input_len = inputs["input_ids"].shape[1]
                out = gen_model.generate(**inputs, **self.gen_kwargs)
                # Composed SpeechLLM generates from inputs_embeds -> output is already only
                # the new tokens; the input_ids path returns prompt+continuation.
                new_tokens = out if self._composed else out[:, input_len:]
                predictions.extend(
                    self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                )

        if model_was_training:
            gen_model.train()

        references = self._extract_references(self.records, self.config.data)
        if self.answer_after is not None or self.answer_regex is not None:
            predictions = [self._extract_answer(p) for p in predictions]
            references = [self._extract_answer(r) for r in references]
        result = self.metric_fn(predictions, references)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as f:
                for pred, ref in zip(predictions, references):
                    row: dict[str, Any] = {"prediction": pred, "reference": ref}
                    if step is not None:
                        row["step"] = step
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return {f"{prefix}_{k}": v for k, v in result.items()}


class _GenerationEvalCallback(TrainerCallback):
    """Run a :class:`GenerationEvaluator` from a trainer hook and log its metrics.

    Metrics go through the bound trainer (:meth:`bind_trainer`), which also records
    them in ``state.log_history``; without one they are appended there directly.
    """

    #: Metric-key prefix and the predictions file written under ``output_dir``.
    prefix: str = "eval"

    def __init__(self, evaluator: GenerationEvaluator):
        self.evaluator = evaluator
        self.trainer: Any = None

    def bind_trainer(self, trainer) -> None:
        """Log metrics through ``trainer`` (call once the trainer is constructed)."""
        self.trainer = trainer

    def _evaluate_and_log(self, args, state, model) -> None:
        if model is None:
            return
        # Write JSONL only from global rank 0; local_rank would fire once per node.
        save_path = None
        if state.is_world_process_zero:
            save_path = os.path.join(args.output_dir, f"{self.prefix}_predictions.jsonl")
        metrics = self.evaluator.evaluate(
            model, save_path=save_path, step=state.global_step, prefix=self.prefix
        )
        if not metrics:
            return
        # Trainer.log() already appends to state.log_history; only append manually on
        # the no-trainer path, else every metric lands in trainer_state.json twice.
        if self.trainer is not None:
            self.trainer.log(metrics)
        else:
            state.log_history.append({**metrics, "step": state.global_step})
            _log.info(f"{self.prefix}-generation: step {state.global_step}: {metrics}")


class GenerateEvalCallback(_GenerationEvalCallback):
    """Score the validation records by AR decoding at every evaluation."""

    prefix = "eval"

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        self._evaluate_and_log(args, state, model)


class TestEvalCallback(_GenerationEvalCallback):
    """Score the held-out test records by AR decoding once, at the end of training."""

    prefix = "test"

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._evaluate_and_log(args, state, model)
