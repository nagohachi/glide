# Tutorial: LLM GRPO (and GSPO)

Reinforcement learning with Group Relative Policy Optimization. Running the same
config with `glide gspo` switches to sequence-level importance sampling (GSPO).

## 1. Data

Each record needs a `prompt` plus any columns your reward functions read:

```json
{"prompt": "What is 2+2? Reason step by step.", "reference": "4"}
```

## 2. Config

`configs/my_grpo.yaml`:

```yaml
extends: ../base.yaml
task: grpo
modality: text

model:
  model_name_or_id: Qwen/Qwen3-1.7B

data:
  train_jsonl_path: data/rl_train.jsonl
  eval_jsonl_path: data/rl_eval.jsonl

rl:
  rewards:
    - name: format                       # reward a <think>...</think> block
      weight: 0.2
    - name: exact_match                  # reward matching the `reference` column
      weight: 1.0
      kwargs: { reference_key: reference }

training:
  output_dir: outputs/my_grpo
  learning_rate: 1.0e-6
  num_generations: 8                     # group size G
  max_completion_length: 512
  per_device_train_batch_size: 8
```

## 3. Train

```bash
glide grpo configs/my_grpo.yaml           # GRPO (token-level IS)
glide gspo configs/my_grpo.yaml           # GSPO (sequence-level IS)
```

## Custom rewards

Drop a plugin in your project `src/` and reference it:

```python
# src/my_rewards.py
from glide.registry import rewards

@rewards.register("digits_only")
def build_digits_only():
    import re
    def _reward(prompts=None, completions=None, **cols):
        from glide.plugins.rewards import completion_text
        return [1.0 if re.fullmatch(r"\d+", completion_text(c).strip()) else 0.0
                for c in completions]
    return _reward
```

```yaml
plugins: ["src/my_rewards.py"]
rl:
  rewards:
    - { name: digits_only, weight: 0.5 }
```

## vLLM rollout

For faster generation, enable vLLM in `rl:` and run a TRL vLLM server:

```yaml
rl:
  use_vllm: true
  vllm_mode: server
training:
  vllm_server_host: 127.0.0.1
```
