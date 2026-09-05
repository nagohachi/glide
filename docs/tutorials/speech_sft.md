# Tutorial: Speech SFT (ASR / AST)

Fine-tune a **composed Speech-LLM** — an audio **encoder** + a **projector** + a text
**LLM** — for recognition or translation, with autoregressive CER/WER/BLEU eval and
train/eval token-accuracy logging.

## 1. Data

JSONL with an audio path and the target transcript in `text` (the instruction lives
in the config, not per-record). An optional `speed` field drives runtime perturbation:

```json
{"audio": "/data/utt001.wav", "text": "こんにちは世界", "speed": 1.0}
```

The `audio` field may be a path or a `{"array": [...], "sampling_rate": 16000}` object.

## 2. Config

A composed model is `encoder + projector + LLM`, each chosen independently. Example
(`configs/examples/qwen3asr_csj10h_sft.yaml` is the full version):

```yaml
extends: [../data_root.yaml, ../base.yaml]
task: sft
modality: speech

model:
  model_name_or_id: Qwen/Qwen3-1.7B               # the text LLM
  attn_implementation: sdpa
  torch_dtype: bfloat16

special_tokens:
  audio_token: "<audio>"             # marker spliced with projected audio frames
  additional: ["<audio>"]

data:
  root_key: csj                        # data_roots[root_key] from data_root.yaml
  train_jsonl_path: native_sft_10h/train.jsonl  # relative to the data root
  eval_jsonl_path: native_sft/dev.jsonl
  response_field: text
  reference_field: text
  max_eval_samples: 200              # cap eval (AR generation is slow)

speech:
  enabled: true
  task: recognition                  # or `translation`
  sample_rate: 16000
  length_grouped_sampler: true       # similar-length batches, reshuffled per epoch
  max_tokens_per_batch: 960000       # dynamic batching: ~audio samples/device-batch
  encoder:
    name: qwen3_asr_aut              # built-in: whisper | wavlm | xls_r | qwen3_asr_aut | qwen_omni_aut
    pretrained: Qwen/Qwen3-ASR-1.7B
    freeze: false
  projector:
    name: identity                  # built-in: identity | mlp_gelu | qwen3_asr_proj
  augment:
    speed_perturb: { enabled: true, from_field: true, field_name: speed }

template:
  system_prompt: "Please transcribe the speech to text."   # the instruction (system turn)
  train_on_completions_only: true   # masks the chat-template prompt prefix (no marker needed)

eval:
  generate: { enabled: true }
  metrics: [cer, wer]               # for translation add: bleu, rouge

training:
  output_dir: outputs/asr_sft
  per_device_train_batch_size: 48   # cap; max_tokens_per_batch is the real bound
  learning_rate: 1.0e-5
  eval_strategy: steps
  eval_steps: 500

distributed:
  nproc_per_node: null              # null = use all visible GPUs (no env vars)
```

## 3. Train

```bash
glide sft configs/examples/qwen3asr_csj10h_sft.yaml          # 1 GPU
```

Multi-GPU is driven by the YAML `distributed.nproc_per_node` (no `NPROC_PER_NODE`
env var): `glide` self-launches `torch.distributed.run` across that many GPUs. On a
cluster with `wait_gpu`, point it at the glide binary + config directly:

```bash
wait_gpu 6 --server sacs01 --name asr_sft \
  /path/to/glide/.venv/bin/glide sft /path/to/glide/configs/examples/qwen3asr_csj10h_sft.yaml
```

## How it works

- **`ComposedSpeechCollator`** renders `system=instruction`, `user=<audio>`,
  `assistant=target`, extracts audio features, and masks the chat-template prompt
  prefix (completion-only loss; model-agnostic, no `response_template` needed).
- **`SpeechLLM`** encodes audio, projects it, and **splices** the frames into the LLM
  input embeddings at the `<audio>` marker; audio positions are excluded from the loss.
- The **length-grouped batch sampler** forms similar-length batches under the
  `max_tokens_per_batch` budget (per-rank-even for DDP) and reshuffles each epoch.
- At each eval, teacher-forced `eval_loss` + `eval_mean_token_accuracy` and AR-decoded
  `eval_cer`/`eval_wer` are logged (train `mean_token_accuracy` logs every step).

## Thinking-mode SFT (Qwen3)

Train the model to emit a reasoning block before the answer. Set `enable_thinking`
and, when the reference text contains the reasoning, strip it before scoring:

```yaml
template:
  enable_thinking: true             # supervises <think>...</think> + transcript
eval:
  answer_after: "</think>"          # score only the transcript after the think block
```

If reasoning is a separate field, set `data.reasoning_field`; glide passes it as the
chat template's `reasoning_content`. See `qwen3asr_csj10h_think_sft.yaml`.

## Other encoders / projectors

Built-ins cover Whisper / WavLM / XLS-R / Qwen3-ASR AuT / Qwen-Omni AuT and
`identity` / `mlp_gelu` / pretrained Qwen projectors. To add your own, register a
plugin and name it in the config:

```python
# src/my_encoder.py
from glide.registry import audio_encoders

@audio_encoders.register("my_enc")
def build(cfg, sample_rate): ...
```

```yaml
plugins: ["src/my_encoder.py"]
speech:
  encoder: { name: my_enc }
```
