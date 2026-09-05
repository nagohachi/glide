# Tutorial: LLM SFT

Supervised fine-tuning of a text LLM with completion-only loss and sequence
packing.

## 1. Data

JSONL, one record per line. Either conversational:

```json
{"messages": [{"role": "user", "content": "Translate to French: hello"}, {"role": "assistant", "content": "bonjour"}]}
```

or prompt/response:

```json
{"prompt": "Translate to French: hello", "response": "bonjour"}
```

## 2. Config

`configs/my_sft.yaml`:

```yaml
extends: ../base.yaml          # if base.yaml is alongside; or use glide's configs/base.yaml
task: sft
modality: text

model:
  model_name_or_id: Qwen/Qwen3-1.7B
  attn_implementation: sdpa     # or flash_attention_2 with the flash-attn extra

data:
  train_jsonl_path: data/train.jsonl
  eval_jsonl_path: data/eval.jsonl

template:
  train_on_completions_only: true   # mask the prompt; train on the answer only
  max_length: 4096

packing:
  enabled: true                     # pack multiple samples per sequence (FFD)

logging:
  report_to: [wandb, tensorboard]
  project: my-sft

training:
  output_dir: outputs/my_sft
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
```

## 3. Train

```bash
glide sft configs/my_sft.yaml
# override anything on the CLI:
glide sft configs/my_sft.yaml --training.learning_rate 2e-5 --peft.enabled true
```

Outputs land in `outputs/my_sft/v{N}-{datetime}/`, alongside `glide_config.yaml`
(the exact resolved config). LoRA is one flag away (`--peft.enabled true`).

## How it works

`glide` converts each record to a conversational `{"prompt", "completion"}` pair
and hands it to TRL's `SFTTrainer`, which applies the chat template, masks the
prompt (completion-only loss), and packs sequences — all configured from your YAML.
