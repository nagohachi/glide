# glide

A **TRL-based post-training library** specialized in:

- **LLM post-training** — SFT and RL (GRPO, GSPO).
- **Speech LLM** (speech + text input) — recognition & translation, with
  autoregressive decoding and **WER / CER / BLEU / ROUGE** at validation time.
- **Vision LLM** (vision + text input) — model-agnostic via
  `AutoModelForImageTextToText`, with AR-decoding evaluation.

Everything is driven by composable YAML (`base.yaml` + overrides), runs through
TRL trainers, and is installed into your project environment and extended with
plugins (custom audio encoders, projectors, reward functions, metrics).

---

## Installation

`glide` uses [uv](https://docs.astral.sh/uv/) with two **mutually-exclusive** CUDA
builds selected by extras:

```bash
# CUDA 12.6 (system default) -> the default ./.venv
uv sync --extra cu126 --extra dev

# CUDA 13.0 -> switch the toolkit, then sync the cu130 build
module switch cuda/13.0
uv sync --extra cu130 --extra dev
```

### Using both CUDA builds at once

A single `.venv` holds exactly **one** resolution — `cu126` and `cu130` are
declared as conflicting extras (torch cannot be both at once). To keep both
available simultaneously, give each its own environment directory:

```bash
uv sync --extra cu126 --extra dev                                    # -> .venv       (cu126)
UV_PROJECT_ENVIRONMENT=.venv-cu130 uv sync --extra cu130 --extra dev # -> .venv-cu130  (cu130)

uv run --no-sync python train.py                                     # cu126
UV_PROJECT_ENVIRONMENT=.venv-cu130 uv run --no-sync python train.py  # cu130 (after `module switch cuda/13.0`)
```

### Flash-Attention (optional)

Flash-Attention has no prebuilt wheel for bleeding-edge torch, so it compiles from
source (~30–60 min) and **cannot** be installed by `uv sync`. Install it *after*
the torch extra, with `--no-build-isolation`. On uv-managed Python you must force
`CC=gcc CXX=g++` — the managed interpreter's `sysconfig` points at `clang`, which
is absent on the GPU hosts:

```bash
uv pip install --python .venv/bin/python ninja packaging setuptools wheel
CC=gcc CXX=g++ MAX_JOBS=4 \
  uv pip install --python .venv/bin/python flash-attn --no-build-isolation
# then set model.attn_implementation: flash_attention_2 in your config
```

Verified: flash-attn **2.8.3.post1** against torch 2.11+cu126 (functional check passed).

System **ffmpeg 7.1** is used for audio decoding (via `librosa`/`soundfile`).

---

## Using glide from your project

`glide` is a standard installable package, so install it however you like — there
is no required directory layout. Editable install from a checkout:

```bash
uv pip install -e /path/to/glide     # or: pip install -e, uv add, a regular install

# your custom code (plugins) lives in src/, your configs in configs/
glide sft  configs/my_sft.yaml
glide grpo configs/my_grpo.yaml --model.name Qwen/Qwen3-1.7B --training.learning_rate 1e-6
```

A common per-project pattern is to vendor it under `libs/` (e.g.
`git clone <glide-url> libs/glide && uv pip install -e libs/glide`) so the library
is pinned alongside one experiment — but that's just a convention; nothing in glide
depends on it (plugin paths resolve against your working directory, not `libs/`).

`output_dir` from the YAML is versioned automatically to
`output_dir/v{N}-{datetime}`, and the fully-resolved config is snapshotted there
as `glide_config.yaml`.

---

## CLI

```
glide sft   [config.yaml] [--dotted.key value ...]   # supervised fine-tuning
glide grpo  [config.yaml] ...                         # GRPO
glide gspo  [config.yaml] ...                         # GSPO (sequence-level IS)
glide eval  [config.yaml] [-c checkpoint] ...         # AR-decoding eval on data.eval
glide test  [config.yaml] [-c checkpoint] [-o preds.jsonl]  # held-out test on data.test
glide docs  [-o docs/api] [--serve]                   # API docs from docstrings
```

Any `--a.b.c value` after the config path overrides the YAML (highest precedence).

### Multi-GPU / distributed

Distributed launch is configured in **YAML** (not env vars). `glide <task>`
re-launches itself under `torch.distributed.run` (DDP) when `nproc_per_node`
resolves to > 1 — no need to call `torchrun` yourself:

```yaml
distributed:
  nproc_per_node: null   # null = auto = all visible GPUs; or an integer
  nnodes: 1              # multi-node: set nnodes / node_rank / master_addr / master_port
```

```bash
glide sft configs/my_sft.yaml            # uses all visible GPUs automatically
glide sft configs/my_sft.yaml --distributed.nproc_per_node 4   # or override on the CLI
```

`nproc_per_node` defaults to the visible GPU count and is clamped to it if set
higher. Gradient checkpointing defaults to non-reentrant so it works under DDP.

---

## Configuration

`base.yaml` holds defaults; run configs `extends:` it and override only what
differs. See [`configs/`](configs/). The schema is the dataclass tree in
[`glide.config.schema`](src/glide/config/schema.py); the `training:` block is
forwarded verbatim to the matching TRL config (`SFTConfig`/`GRPOConfig`).

### Special tokens (from YAML)

```yaml
special_tokens:
  additional: ["<audio>", "<audio_pad>", "<image>", "<image_pad>"]
  audio_token: "<audio>"
  audio_pad_token: "<audio_pad>"
  resize_embeddings: true
  pad_to_multiple_of: 8
```

New embedding rows are mean-initialized; the vocab is padded to a kernel-friendly
multiple.

---

## Reinforcement learning

`glide grpo` / `glide gspo` build a TRL `GRPOTrainer` (GSPO sets
`importance_sampling_level="sequence"`). Reward functions are composed with weights:

```yaml
rl:
  rewards:
    - { name: format, weight: 0.2 }
    - { name: cer,    weight: 1.0, kwargs: { reference_key: reference } }
```

Built-in rewards: `format`, `length`, `exact_match`, `cer`. Add your own via the
[plugin API](#plugins).

---

## Plugins

Put plugins anywhere in your project (e.g. `src/`) and list them in the config —
paths resolve against your working directory, no `sys.path` hacks required:

```yaml
plugins: ["src/my_rewards.py", "src.my_encoder"]
```

```python
# src/my_rewards.py
from glide.registry import rewards

@rewards.register("my_reward")
def build_my_reward(scale: float = 1.0):
    def _reward(prompts=None, completions=None, **columns):
        return [scale * len(c) for c in completions]
    return _reward
```

Registries: `rewards`, `metrics`, `audio_encoders`, `projectors`, `templates`,
`collators`, `models`. Subclass `glide.models.AudioEncoder` / `Projector` for
custom multimodal components.

---

## Evaluation & testing

### During training

Set `data.eval` and `eval.generate.enabled: true` to run autoregressive decoding
at each evaluation step. Metrics are logged to the trainer (stdout, wandb, tensorboard)
and per-sample predictions are saved to `{output_dir}/eval_predictions.jsonl`
(overwritten each run so only the latest checkpoint's output is kept):

```yaml
data:
  train: data/train.jsonl
  eval:  data/dev.jsonl
  test:  data/test.jsonl   # evaluated once at the very end of training

eval:
  generate:
    enabled: true
    max_new_tokens: 256
    num_beams: 1           # 1 = greedy; increase for beam search
    batch_size: 8
  metrics: [cer, wer]      # any of: wer, cer, bleu, rouge
  normalize_text: true
```

At the end of training, `data.test` is evaluated once and saved to
`{output_dir}/test_predictions.jsonl` with `test_*` metric keys.

### Standalone evaluation

Run on a validation set from any checkpoint:

```bash
glide eval configs/my_sft.yaml -c outputs/v1-20250601/checkpoint-1000
```

### Standalone test evaluation

Run on the held-out test set, optionally saving per-sample predictions:

```bash
glide test configs/my_sft.yaml -c outputs/v1-20250601/checkpoint-1000
glide test configs/my_sft.yaml -c outputs/v1-20250601/checkpoint-1000 -o preds.jsonl
```

Both commands accept any `--dotted.key value` override, so you can point at a
different test file without editing the YAML:

```bash
glide test configs/my_sft.yaml -c path/to/checkpoint --data.test other_test.jsonl
```

---

## Documentation & tutorials

```bash
glide docs -o docs/api        # generate HTML API docs from docstrings (pdoc)
glide docs --serve             # live-reload at http://localhost:8080
```

Tutorials in [`docs/tutorials/`](docs/tutorials/): LLM-SFT, LLM-GRPO, Speech-SFT,
Speech-GRPO. Online: [nagohachi.github.io/glide](https://nagohachi.github.io/glide/).

---

## Unit & integration tests

CPU and GPU tests are separable:

```bash
uv run --no-sync pytest -m "not gpu"            # fast CPU unit/integration suite
uv run --no-sync pytest -m "slow and not gpu"   # CPU end-to-end (tiny model)
uv run --no-sync pytest -m gpu                  # GPU training/generation tests
```
