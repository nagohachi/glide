"""An ``SFTTrainer`` subclass that uses a length-grouped *batch* sampler.

For the speech modality the spec requires sorting samples by length and forming
batches of similar length, with the order varying each epoch. HF's built-in
``group_by_length`` cannot do this for raw (non-tokenized) multimodal records, so
this trainer injects an :class:`~glide.data.sampler.LengthGroupedBatchSampler`
into the training dataloader and reshuffles it every epoch.
"""

from typing import cast

from torch.utils.data import DataLoader, Dataset
from transformers import TrainerCallback
from trl import SFTTrainer

from ..config.schema import ProjectorWarmupConfig
from ..data.sampler import LengthGroupedBatchSampler

from ..logging_utils import get_logger

_log = get_logger("trainers")

__all__ = ["LengthGroupedSFTTrainer", "projector_warmup_lambdas", "_EpochShuffleCallback"]


def projector_warmup_lambdas(*, projector_only_steps: int, num_training_steps: int,
                             target_lr: float, projector_lr: float,
                             full_warmup_steps: int = 0, warmup_ratio: float = 0.0):
    """Return ``(proj_lambda, rest_lambda)`` LR multipliers for the 2-phase schedule.

    Multipliers are relative to an optimizer base LR of ``projector_lr``:

    * phase 1 (``step < P``): proj ramps 0->1 (i.e. 0->projector_lr), rest = 0;
    * phase 2 (``step >= P``): both ramp 0->``ratio`` (=target_lr/projector_lr) over
      ``W`` steps, then linearly decay to 0 by ``num_training_steps``.
    """
    P, T = projector_only_steps, num_training_steps
    ratio = target_lr / projector_lr
    W = full_warmup_steps if full_warmup_steps > 0 else round(warmup_ratio * (T - P))
    W = max(1, min(W, max(1, T - P)))

    def phase2(step):
        s = step - P
        if s < W:
            return ratio * (s / W)
        return ratio * max(0.0, (T - P - s) / max(T - P - W, 1))

    def proj_lam(step):
        return (step / max(P, 1)) if step < P else phase2(step)

    def rest_lam(step):
        return 0.0 if step < P else phase2(step)

    return proj_lam, rest_lam


class _EpochShuffleCallback(TrainerCallback):
    """Call ``set_epoch`` on the batch sampler at the start of each epoch."""

    def __init__(self, batch_sampler: LengthGroupedBatchSampler):
        self.batch_sampler = batch_sampler

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.batch_sampler.set_epoch(int(state.epoch or 0))


class LengthGroupedSFTTrainer(SFTTrainer):
    """``SFTTrainer`` that batches the training set by length (speech path).

    Pass ``glide_lengths`` (per-sample lengths) and ``glide_batch_size`` via
    ``kwargs``; everything else behaves like a normal ``SFTTrainer``.
    """

    def __init__(self, *args, glide_lengths=None, glide_batch_size=None,
                 glide_max_tokens=None, glide_mega_batch_mult=50, glide_seed=0,
                 glide_warmup: ProjectorWarmupConfig | None = None,
                 glide_target_lr: float | None = None, glide_warmup_ratio: float = 0.0,
                 glide_composed: bool = False, **kwargs):
        self._glide_lengths = glide_lengths
        self._glide_batch_size = glide_batch_size
        self._glide_max_tokens = glide_max_tokens
        self._glide_mega_batch_mult = glide_mega_batch_mult
        self._glide_seed = glide_seed
        #: Two-phase projector-warmup schedule (ProjectorWarmupConfig or None).
        self._glide_warmup = glide_warmup
        self._glide_target_lr = glide_target_lr
        self._glide_warmup_ratio = glide_warmup_ratio
        #: Composed SpeechLLM splices audio embeddings, changing the sequence length,
        #: so TRL's token-aligned entropy metric doesn't apply -> use the plain HF loss.
        self._glide_composed = glide_composed
        self._glide_batch_sampler = None
        super().__init__(*args, **kwargs)
        # SpeechLLM.forward has a **kwargs signature, so HF Trainer sets
        # model_accepts_loss_kwargs=True and (a) skips the loss/gradient_accumulation
        # division in training_step and (b) multiplies by num_processes under DDP -- but
        # the inner LLM returns a *mean* CE loss, not a sum normalized by num_items, so
        # gradients end up ~GA*(N) too large. Force it off for the composed path (HF's own
        # docstring recommends this for mean-loss models). The built-in-tower path keeps
        # the default: those are real HF models that consume num_items_in_batch correctly.
        if self._glide_composed:
            self.model_accepts_loss_kwargs = False
        if glide_lengths is not None:
            # Stripe across ranks IN the sampler (ESPnet-style): each rank gets an
            # equal number of batches every epoch (the sampler truncates to a multiple
            # of num_replicas), so DDP never desyncs at an epoch boundary. We therefore
            # must NOT let accelerate re-shard the dataloader (see get_train_dataloader).
            self._glide_batch_sampler = LengthGroupedBatchSampler(
                glide_lengths,
                batch_size=glide_batch_size or self.args.per_device_train_batch_size,
                max_tokens=glide_max_tokens,
                mega_batch_mult=glide_mega_batch_mult,
                seed=glide_seed,
                drop_last=self.args.dataloader_drop_last,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
            )
            self.add_callback(_EpochShuffleCallback(self._glide_batch_sampler))

    # ------------------------------------------------------------------ #
    # Two-phase: projector-only warmup -> full fine-tuning (DDP-safe).
    # ------------------------------------------------------------------ #
    def _warmup_active(self) -> bool:
        return bool(self._glide_warmup and self._glide_warmup.projector_only_steps > 0)

    def _is_projector_param(self, name: str, warmup: ProjectorWarmupConfig) -> bool:
        return any(pat in name for pat in warmup.projector_patterns)

    def create_optimizer(self, model=None):
        # Note: don't forward `model` to super() -- transformers 4.57 defines
        # create_optimizer(self) (no model arg); 5.x adds an optional one. Calling
        # with no args works on both.
        if not self._warmup_active() or self.optimizer is not None:
            return super().create_optimizer()
        from torch.optim import AdamW

        assert self._glide_warmup is not None and self.model is not None
        w = self._glide_warmup
        proj, rest = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (proj if self._is_projector_param(n, w) else rest).append(p)
        if not proj:
            raise RuntimeError(
                "projector warmup matched 0 projector params; check "
                f"speech.warmup.projector_patterns={w.projector_patterns}"
            )
        # Optimizer base LR = phase-1 projector LR; the per-group scheduler scales
        # phase 2 down to the target LR (ratio = target_lr / projector_lr).
        optimizer = AdamW(
            [{"params": proj}, {"params": rest}],
            lr=w.projector_lr,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )
        self.optimizer = optimizer
        if self.args.process_index == 0:
            _log.info(f"warmup: projector group: {sum(p.numel() for p in proj):,} params | "
                  f"rest: {sum(p.numel() for p in rest):,} params", flush=True)
        return optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if not self._warmup_active():
            return super().create_scheduler(num_training_steps, optimizer)
        from torch.optim.lr_scheduler import LambdaLR

        assert self._glide_warmup is not None
        opt = optimizer or self.optimizer
        assert opt is not None
        w = self._glide_warmup
        target_lr = self._glide_target_lr if self._glide_target_lr is not None else self.args.learning_rate
        proj_lam, rest_lam = projector_warmup_lambdas(
            projector_only_steps=w.projector_only_steps,
            num_training_steps=num_training_steps,
            target_lr=target_lr,
            projector_lr=w.projector_lr,
            full_warmup_steps=w.full_warmup_steps,
            warmup_ratio=self._glide_warmup_ratio,
        )
        if self.args.process_index == 0:
            _log.info(f"warmup: P={w.projector_only_steps} (proj 0->{w.projector_lr:g}), "
                  f"then 0->{target_lr:g}, decay to 0 by {num_training_steps}", flush=True)
        scheduler = LambdaLR(opt, [proj_lam, rest_lam])
        self.lr_scheduler = scheduler
        return scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Composed SpeechLLM changes sequence length (audio splicing); TRL's entropy
        # metric (per_token_entropy * attention_mask) assumes input/logit lengths match,
        # so fall back to transformers' plain loss (the LLM computes it over spliced labels).
        if not self._glide_composed:
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
        from transformers import Trainer

        result = Trainer.compute_loss(self, model, inputs, return_outputs=return_outputs,
                                      num_items_in_batch=num_items_in_batch)
        # Token accuracy: SpeechLLM.forward stashed correct/total counts over the spliced
        # labels; gather across ranks and feed TRL's metric buffer so log() emits
        # mean_token_accuracy (train) / eval_mean_token_accuracy (eval).
        unwrapped = self.accelerator.unwrap_model(model)
        c = getattr(unwrapped, "_tok_correct", None)
        t = getattr(unwrapped, "_tok_total", None)
        if c is not None and t is not None:
            mode = "train" if model.training else "eval"
            c = self.accelerator.gather_for_metrics(c)
            t = self.accelerator.gather_for_metrics(t)
            tot = t.sum()
            self._metrics[mode]["mean_token_accuracy"].append(
                (c.sum() / tot).item() if tot > 0 else 0.0)
        return result

    def get_train_dataloader(self) -> DataLoader:
        if self._glide_batch_sampler is None:
            return super().get_train_dataloader()

        assert self.train_dataset is not None  # batch sampler implies a train set
        # On this path train_dataset is always a torch Dataset (the multimodal
        # JsonlDataset), not an HF datasets.Dataset; narrow the Trainer's union.
        train_dataset = cast(Dataset, self.train_dataset)
        dataloader = DataLoader(
            train_dataset,
            batch_sampler=self._glide_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        # The sampler already strides per-rank (equal batch count), so DON'T let
        # accelerate shard the dataloader -- that would shard again and re-introduce the
        # per-rank imbalance that deadlocks DDP at the epoch boundary. Return the raw
        # loader; the Trainer moves each batch to the device in `_prepare_inputs`.
        return dataloader
