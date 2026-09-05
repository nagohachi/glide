# Tutorial: Speech GRPO (ASR error-correction)

A common, effective form of "Speech RL" is **ASR error-correction with GRPO**: the
policy is a text LLM that rewrites an ASR hypothesis, and the reward is `1 - CER`
against the reference transcript. Because the model input/output is text, this runs
through `glide`'s text GRPO path with a CER reward — no audio in the RL loop.

## 1. Data

```json
{"prompt": "Correct the ASR output.\nHypothesis: helo wrld\nCorrected:", "reference": "hello world"}
```

You can also include a chain-of-thought target style and reward its format.

## 2. Config

`configs/asr_correction_grpo.yaml`:

```yaml
extends: ../base.yaml
task: grpo
modality: text                      # text policy over ASR hypotheses

model:
  model_name_or_id: Qwen/Qwen3-1.7B

data:
  train_jsonl_path: data/correction_train.jsonl
  eval_jsonl_path: data/correction_eval.jsonl
  reference_field: reference

rl:
  rewards:
    - name: cer                     # reward = 1 - CER(correction, reference)
      weight: 1.0
      kwargs: { reference_key: reference }
    - name: format                  # optionally reward a <think>...</think> trace
      weight: 0.2
  response_prefix: "<think>\n"      # optional CoT opener (if your TRL supports prefill)

training:
  output_dir: outputs/asr_correction_grpo
  learning_rate: 1.0e-6
  num_generations: 8
  max_completion_length: 256
```

## 3. Train

```bash
glide grpo configs/asr_correction_grpo.yaml
# sequence-level importance sampling:
glide gspo configs/asr_correction_grpo.yaml
```

## Full speech-in-the-loop RL

To do RL where the model *consumes audio* during rollout, generate with a vLLM
rollout server that accepts audio and register a custom reward that scores the
decoded transcript (e.g. the built-in `cer` reward). Enable `rl.use_vllm: true` and
provide the rollout endpoint via the `training:` vLLM fields. The text path above
covers the most common production case (post-ASR correction) without that
machinery.
