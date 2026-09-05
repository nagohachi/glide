"""Speech-in-the-loop GSPO/GRPO trainer (audio in the rollout).

The text TRL ``GRPOTrainer`` cannot do speech RL: it tokenizes text prompts and
generates with ``model(input_ids=...)`` / vLLM, neither of which feeds audio into a
composed :class:`~glide.models.speech_llm.SpeechLLM` (audio embeds are spliced into
the LLM ``inputs_embeds`` at the ``<audio>`` marker). This module implements a
small, self-contained loop instead:

1. **Rollout** -- sample ``G`` completions per audio prompt (the GRPO/GSPO *group*).
2. **Reward** -- score each completion (default ``1 - CER`` vs the reference), then
   group-normalise to advantages ``A = (r - mean) / (std + eps)``.
3. **Policy update** -- a grad-enabled forward over ``[prompt + completion]`` (with
   audio) yields completion-token log-probs; the loss is the clipped GSPO surrogate
   (sequence-level importance ratio) or GRPO (token-level).

Rollout is behind a pluggable :class:`RolloutBackend`:

* :class:`VLLMRolloutBackend` -- generates through a vLLM server on a **separate
  GPU** and receives fresh policy weights after each optimizer step (production
  path; requires a vLLM build that can serve the composed model -- see
  ``docs/gspo_speech_rollout_plan.md``).
* :class:`LocalRolloutBackend` -- generates with the in-process model via HF
  ``generate``. Used to validate the GSPO math without a server; **not** the
  default when ``rl.use_vllm`` is set.

GSPO == GRPO with a *sequence-level* importance ratio; the task (``gspo`` vs
``grpo``) selects which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import transformers

from ..config.schema import GlideConfig, Task
from ..config.loader import build_training_args
from ..data.build import build_multimodal_dataset
from ..data.composed_collator import ComposedSpeechCollator
from ..models.loader import load_model_and_processor
from .common import build_reward_funcs, init_plugins

__all__ = ["build_speech_rl_trainer", "SpeechGSPOTrainer"]


# --------------------------------------------------------------------------- #
# Rollout backends
# --------------------------------------------------------------------------- #
@dataclass
class _RolloutCtx:
    """Shared handles a backend needs to roll out (set by the trainer)."""

    collator: ComposedSpeechCollator
    tokenizer: Any
    num_generations: int
    temperature: float
    top_p: float
    max_new_tokens: int
    pad_token_id: int
    eos_token_id: int | None


class RolloutBackend:
    """Generate ``G`` completions per record and (optionally) sync weights to a server.

    ``rollout`` returns, per input record, a dict::

        {"prompt_ids": LongTensor(P),         # prompt tokens incl. the <audio> marker
         "audio": {feature tensors (1, ...)}, # the record's encoder inputs
         "comps": [{"ids": LongTensor(Lc),     # completion tokens
                    "text": str,
                    "old_logp": Tensor(Lc) | None}]}  # behaviour-policy token logps

    ``old_logp`` is ``None`` when the rollout policy *is* the training policy this
    step (on-policy); the trainer then uses ``new_logp.detach()`` (ratio == 1).
    """

    def bind(self, ctx: _RolloutCtx) -> None:
        self.ctx = ctx

    def rollout(self, model, records: list[dict]) -> list[dict]:
        raise NotImplementedError

    def sync_weights(self, model) -> None:
        """Push updated policy weights to the rollout engine (no-op if in-process)."""


class LocalRolloutBackend(RolloutBackend):
    """In-process rollout via :meth:`SpeechLLM.generate` (batch-1 per record).

    ``generation_inputs`` right-pads (``_splice`` re-pads), which breaks batched
    generation across records, so we generate one record at a time with
    ``num_return_sequences=G``. Correct and simple; slower than a vLLM server.
    """

    @torch.no_grad()
    def rollout(self, model, records: list[dict]) -> list[dict]:
        ctx = self.ctx
        device = next(model.parameters()).device
        out: list[dict] = []
        for rec in records:
            gi = ctx.collator.generation_inputs([rec])
            gi = {k: v.to(device) for k, v in gi.items()}
            prompt_ids = gi["input_ids"][0].detach().to("cpu")
            audio = {k: v for k, v in gi.items()
                     if k not in ("input_ids", "attention_mask")}
            gen = model.generate(
                **gi,
                do_sample=True,
                temperature=ctx.temperature,
                top_p=ctx.top_p,
                num_return_sequences=ctx.num_generations,
                max_new_tokens=ctx.max_new_tokens,
                pad_token_id=ctx.pad_token_id,
            )  # (G, Lgen) -- new tokens only (generate from inputs_embeds)
            comps = []
            for j in range(gen.shape[0]):
                ids = _truncate_at_eos(gen[j].detach().to("cpu"), ctx.eos_token_id,
                                       ctx.pad_token_id)
                text = ctx.tokenizer.decode(ids, skip_special_tokens=True)
                comps.append({"ids": ids, "text": text, "old_logp": None})
            out.append({"prompt_ids": prompt_ids,
                        "audio": {k: v.detach() for k, v in audio.items()},
                        "comps": comps})
        return out


# Rename map: training-side composed SpeechLLM param names -> the vLLM AuT server's
# ``thinker.*`` names (the server's load_weights routes those through its mapper).
_GLIDE_TO_THINKER = (
    ("encoder.model.", "thinker.audio_tower."),
    ("llm.model.", "thinker.model."),
)


def _glide_to_thinker(name: str, *, tie_word_embeddings: bool = True) -> str | None:
    """Map a composed-SpeechLLM param name to its rollout-server name (or ``None``).

    The projector and lm_head must NOT be silently dropped: a trainable projector
    (``mlp_gelu`` is the schema default) diverges from the rollout policy otherwise,
    and ``lm_head`` is only redundant with ``embed_tokens`` for *tied* LLMs.
    """
    if name.startswith("llm.lm_head."):
        # Tied embeddings: pushing embed_tokens already covers lm_head. Untied: push it.
        if tie_word_embeddings:
            return None
        return "thinker.lm_head." + name[len("llm.lm_head."):]
    for old, new in _GLIDE_TO_THINKER:
        if name.startswith(old):
            return new + name[len(old):]
    if name.startswith("projector."):
        return name  # served as-is (the composed projector is not part of thinker.*)
    return None  # buffers etc. -> not served


class VLLMRolloutBackend(RolloutBackend):
    """Rollout through the AuT-LALM vLLM server (separate GPU); see scripts/aut_lalm/serve.py.

    Completions (with behaviour-policy logprobs) come from the server over HTTP;
    audio is passed by *path* (shared filesystem). The prompt_ids + audio features
    for the trainer's grad forward are still built locally via the collator -- only
    generation is offloaded. Weight-sync saves the policy (remapped ``glide -> thinker``)
    to a shared safetensors file and tells the server to reload it.
    """

    def __init__(self, host: str, port: int, sync_path: str):
        self.base_url = f"http://{host}:{port}"
        self.sync_path = sync_path

    def _post(self, route: str, payload: dict, timeout: float = 600.0) -> dict:
        import requests

        # Bypass any cluster http(s)_proxy (Squid) -- the rollout server is internal;
        # routing localhost/compute-node traffic through the proxy fails (ERR_CONNECT_FAIL).
        r = requests.post(f"{self.base_url}{route}", json=payload, timeout=timeout,
                          proxies={"http": None, "https": None})
        r.raise_for_status()
        return r.json()

    @torch.no_grad()
    def rollout(self, model, records: list[dict]) -> list[dict]:
        ctx = self.ctx
        device = next(model.parameters()).device
        audio_field = ctx.collator.data.audio_field
        prompt_field = ctx.collator.data.prompt_field
        # Transmit the per-record prompt text and system prompt: the local collator bakes
        # these into prompt_ids for the grad forward, so the rollout server must condition
        # on the SAME context or completions/old_logps come from a different distribution
        # (silently wrong importance ratios and rewards).
        resp = self._post("/rollout", {
            "audio_paths": [rec[audio_field] for rec in records],
            "prompts": [rec.get(prompt_field, "") for rec in records],
            "system_prompt": ctx.collator.template.system_prompt,
            "n": ctx.num_generations,
            "temperature": ctx.temperature,
            "top_p": ctx.top_p,
            "max_tokens": ctx.max_new_tokens,
        })
        rollouts = resp["rollouts"]
        out: list[dict] = []
        for rec, comps_raw in zip(records, rollouts):
            gi = ctx.collator.generation_inputs([rec])
            gi = {k: v.to(device) for k, v in gi.items()}
            prompt_ids = gi["input_ids"][0].detach().to("cpu")
            audio = {k: v.detach() for k, v in gi.items()
                     if k not in ("input_ids", "attention_mask")}
            comps = []
            for c in comps_raw:
                comps.append({
                    "ids": torch.tensor(c["token_ids"], dtype=torch.long),
                    "text": c["text"],
                    "old_logp": torch.tensor(c["logprobs"], dtype=torch.float),
                })
            out.append({"prompt_ids": prompt_ids, "audio": audio, "comps": comps})
        return out

    def sync_weights(self, model) -> None:
        import os

        from safetensors.torch import save_file

        tie = bool(getattr(model.config, "tie_word_embeddings", True))
        sd = {}
        for name, p in model.named_parameters():
            nk = _glide_to_thinker(name, tie_word_embeddings=tie)
            if nk is not None:
                sd[nk] = p.detach().to(torch.bfloat16).cpu().contiguous()
        os.makedirs(os.path.dirname(self.sync_path) or ".", exist_ok=True)
        save_file(sd, self.sync_path)
        self._post("/reload", {"path": self.sync_path})


def _truncate_at_eos(ids: torch.Tensor, eos_id: int | None, pad_id: int) -> torch.Tensor:
    """Keep tokens up to and including the first EOS; drop trailing pad/eos padding."""
    ids = ids.tolist()
    if eos_id is not None and eos_id in ids:
        ids = ids[: ids.index(eos_id) + 1]
    else:
        while ids and ids[-1] == pad_id:
            ids.pop()
    return torch.tensor(ids, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class _SpeechRLCallback(transformers.TrainerCallback):
    """Push weights to the rollout server after the optimizer steps."""

    def __init__(self, trainer: "SpeechGSPOTrainer"):
        self._trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        t = self._trainer
        every = max(1, t.gconf.rl.vllm_sync_every)
        if state.global_step % every == 0:
            t.backend.sync_weights(t.accelerator.unwrap_model(t.model))


class SpeechGSPOTrainer(transformers.Trainer):
    """GRPO/GSPO over a composed Speech-LLM with audio in the rollout."""

    def __init__(self, gconf: GlideConfig, backend: RolloutBackend, reward_funcs,
                 reward_weights, **kwargs):
        self.gconf = gconf
        self.backend = backend
        self.reward_funcs = reward_funcs
        self.reward_weights = reward_weights
        self._sequence_level = gconf.task is Task.GSPO
        self._extra_logs: dict[str, float] = {}
        super().__init__(**kwargs)
        # v1 is single-GPU: _completion_logps calls model.logits_with_audio (a custom
        # method) which DistributedDataParallel does not proxy -> AttributeError at step 1.
        # Fail loudly with a fix instead of crashing cryptically.
        if self.accelerator.num_processes > 1:
            raise NotImplementedError(
                "Speech GSPO/GRPO (rl_speech) is single-GPU only in v1: the grad forward "
                "calls model.logits_with_audio on the DDP-wrapped model, which DDP does not "
                "proxy. Pin distributed.nproc_per_node: 1 and run on a single visible GPU."
            )
        self.add_callback(_SpeechRLCallback(self))

    # Inputs are raw record lists, not tensors -- skip the default device move.
    def _prepare_inputs(self, inputs):
        return inputs

    def get_train_dataloader(self):
        # Default collator would torch-collate dicts; we want the raw record list.
        return super().get_train_dataloader()

    # ---- reward / advantage ------------------------------------------------ #
    def _group_rewards(self, completions_text: list[str], reference) -> list[float]:
        ref_field = self.gconf.data.reference_field
        n = len(completions_text)
        cols = {ref_field: [reference] * n}
        cols.setdefault("reference", [reference] * n)  # alias for the built-in cer reward
        total = [0.0] * n
        for fn, w in zip(self.reward_funcs, self.reward_weights):
            vals = fn(prompts=[""] * n, completions=completions_text, **cols)
            for i, v in enumerate(vals):
                total[i] += float(w) * float(v)
        return total

    @staticmethod
    def _advantages(rewards: list[float], scale: bool) -> torch.Tensor:
        r = torch.tensor(rewards, dtype=torch.float32)
        adv = r - r.mean()
        if scale and r.numel() > 1:
            adv = adv / (r.std(unbiased=False) + 1e-4)
        return adv

    # ---- per-record policy logps ------------------------------------------ #
    def _completion_logps(self, model, prompt_ids: torch.Tensor, comps: list[dict],
                          audio: dict):
        """Return per-completion token log-probs ``list[Tensor(Lc)]`` (grad-enabled).

        Generates the grad forward for one record's whole group at once (shared audio
        + prompt; completions right-padded). Completion tokens are the trailing span
        of each spliced row, located via the post-splice attention mask.
        """
        device = next(model.parameters()).device
        g = len(comps)
        P = prompt_ids.numel()
        lcs = [c["ids"].numel() for c in comps]
        max_c = max(lcs)
        pad_id = self.backend.ctx.pad_token_id

        input_ids = torch.full((g, P + max_c), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((g, P + max_c), dtype=torch.long, device=device)
        prompt_ids = prompt_ids.to(device)
        for i, c in enumerate(comps):
            lc = lcs[i]
            input_ids[i, :P] = prompt_ids
            if lc:
                input_ids[i, P:P + lc] = c["ids"].to(device)
            attn[i, : P + lc] = 1

        audio_b = {k: v.to(device).repeat(g, *([1] * (v.dim() - 1))) for k, v in audio.items()}
        logits, sp_attn = model.logits_with_audio(
            input_ids=input_ids, attention_mask=attn, **audio_b
        )
        # Divide logits by the sampling temperature before scoring: vLLM's old_logps come
        # back at rl.temperature, so plain (T=1) log-softmax here biases the GSPO ratio
        # exp(new-old) for any temperature != 1.0 (TRL applies the same scaling). For the
        # local backend old_logp is None and both sides get this scaling -> ratio == 1.
        temperature = self.backend.ctx.temperature or 1.0
        logp = F.log_softmax(logits.float() / temperature, dim=-1)

        per: list[torch.Tensor] = []
        for i, c in enumerate(comps):
            lc = lcs[i]
            if lc == 0:
                per.append(logits.new_zeros(0))
                continue
            valid = int(sp_attn[i].sum())
            # completion tokens occupy spliced positions [valid-lc, valid); each is
            # predicted by the logits one position earlier.
            pred = logp[i, valid - lc - 1: valid - 1, :]            # (lc, V)
            tok = c["ids"].to(device)                               # (lc,)
            per.append(pred.gather(-1, tok.unsqueeze(-1)).squeeze(-1))
        return per

    # ---- GSPO/GRPO loss for one batch of records --------------------------- #
    def _gspo_loss(self, model, records: list[dict]):
        # Single-GPU v1: the policy is unwrapped, so the custom forward method is
        # reachable and grads still flow. (Multi-GPU DDP would need a forward shim.)
        policy = self.accelerator.unwrap_model(model)
        roll = self.backend.rollout(policy, records)
        rl = self.gconf.rl
        eps = rl.clip_eps

        losses = []
        rew_means, rew_stds, comp_lens, clipfracs, kls = [], [], [], [], []
        for rec, item in zip(records, roll):
            comps = item["comps"]
            ref = rec.get(self.gconf.data.reference_field)
            if ref is None:
                raise ValueError(
                    f"record is missing the reference field "
                    f"{self.gconf.data.reference_field!r} (set data.reference_field to the "
                    "field holding the target text). Without it the CER reward returns 0 for "
                    "every completion, so all advantages are 0 and training silently no-ops."
                )
            rewards = self._group_rewards([c["text"] for c in comps], ref)
            adv = self._advantages(rewards, rl.scale_rewards).to(
                next(model.parameters()).device
            )

            new_tok = self._completion_logps(model, item["prompt_ids"], comps, item["audio"])
            for i, c in enumerate(comps):
                lc = new_tok[i].numel()
                if lc == 0:
                    continue
                new_seq = new_tok[i].sum()
                old_seq = (c["old_logp"].to(new_seq.device).sum()
                           if c["old_logp"] is not None else new_seq.detach())
                if self._sequence_level:                       # GSPO: length-normalised
                    ratio = torch.exp((new_seq - old_seq) / lc)
                    a = adv[i]
                    unclipped = ratio * a
                    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * a
                    losses.append(-torch.min(unclipped, clipped))
                    clipfracs.append(float((unclipped > clipped).float()))
                else:                                          # GRPO: per-token ratio
                    old_tok = (c["old_logp"].to(new_tok[i].device)
                               if c["old_logp"] is not None else new_tok[i].detach())
                    ratio = torch.exp(new_tok[i] - old_tok)
                    a = adv[i]
                    unclipped = ratio * a
                    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * a
                    losses.append(-torch.min(unclipped, clipped).mean())
                    clipfracs.append(float((unclipped > clipped).float().mean()))
                comp_lens.append(lc)
            rew_means.append(sum(rewards) / len(rewards))
            rew_stds.append(float(torch.tensor(rewards).std(unbiased=False)))

        loss = (torch.stack(losses).mean() if losses
                else next(model.parameters()).sum() * 0.0)
        self._extra_logs = {
            "reward_mean": _mean(rew_means),
            "reward_std": _mean(rew_stds),
            "completion_len": _mean(comp_lens),
            "clip_frac": _mean(clipfracs),
        }
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        loss = self._gspo_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)
        return loss.detach()

    def log(self, logs: dict, start_time=None):
        if self._extra_logs:
            logs = {**logs, **self._extra_logs}
        try:
            super().log(logs, start_time)
        except TypeError:  # older transformers signature
            super().log(logs)


def _mean(xs) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build_speech_rl_trainer(config: GlideConfig) -> SpeechGSPOTrainer:
    """Assemble a :class:`SpeechGSPOTrainer` from ``config`` (called by ``build_rl_trainer``)."""
    init_plugins(config)
    if config.peft.enabled:
        # rl.py wires peft only for the text GRPO path; the composed speech loop would
        # otherwise silently full-fine-tune. LoRA here would also break weight-sync
        # (_glide_to_thinker maps base names, not adapter names), so fail loudly.
        raise NotImplementedError(
            "peft.enabled is not supported by the speech GSPO/GRPO loop (rl_speech): "
            "LoRA adapter param names would not map through the vLLM weight-sync. Run "
            "speech RL with peft.enabled=false (full fine-tuning), or use PEFT via the SFT "
            "path and start RL from the merged checkpoint."
        )
    loaded = load_model_and_processor(config)
    reward_funcs, reward_weights = build_reward_funcs(config)
    if not reward_funcs:
        raise ValueError("Speech GSPO/GRPO needs at least one rl.rewards entry "
                         "(e.g. `cer`).")

    tokenizer = loaded.tokenizer
    enc = loaded.model.encoder
    collator = ComposedSpeechCollator(
        tokenizer=tokenizer,
        feature_extractor=enc.feature_extractor,
        input_kind=enc.input_kind,
        audio_token=config.special_tokens.audio_token,
        data=config.data,
        template=config.template,
        sample_rate=config.speech.sample_rate,
        completion_only=config.template.train_on_completions_only,
        train=False,  # no speed-perturb / specaugment during rollout
    )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    ctx = _RolloutCtx(
        collator=collator,
        tokenizer=tokenizer,
        num_generations=config.rl.num_generations,
        temperature=config.rl.temperature,
        top_p=config.rl.top_p,
        max_new_tokens=config.rl.max_completion_length,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if config.rl.use_vllm:
        backend: RolloutBackend = VLLMRolloutBackend(
            config.rl.vllm_server_host, config.rl.vllm_server_port,
            config.rl.vllm_sync_path,
        )
    else:
        backend = LocalRolloutBackend()
    backend.bind(ctx)

    args = build_training_args(config, transformers.TrainingArguments)
    args.remove_unused_columns = False
    # The composed Speech-LLM ties embeddings<->lm_head (shared storage); safetensors
    # refuses shared tensors, so checkpoints must use torch.save (cf. SFT path).
    args.save_safetensors = False

    return SpeechGSPOTrainer(
        gconf=config,
        backend=backend,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        model=loaded.model,
        args=args,
        train_dataset=build_multimodal_dataset(config, "train"),
        data_collator=lambda recs: recs,  # pass the raw record list through
        processing_class=tokenizer,
    )
