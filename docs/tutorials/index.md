# glide tutorials

Hands-on walkthroughs. Each assumes you've installed glide (`uv sync --extra cu126
--extra dev`) and run commands from your project root.

- [LLM SFT](llm_sft.md) — supervised fine-tuning of a text LLM.
- [LLM GRPO](llm_grpo.md) — RL with GRPO/GSPO and reward functions.
- [Speech SFT](speech_sft.md) — ASR/AST fine-tuning with WER/CER/BLEU eval.
- [Speech GRPO](speech_grpo.md) — RL for ASR error-correction (text-side CER reward).

API reference: `glide docs -o docs/api` (generated from docstrings with pdoc).
