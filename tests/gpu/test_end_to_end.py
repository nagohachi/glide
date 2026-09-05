"""End-to-end SFT training tests.

These build a *tiny* randomly-initialized causal LM offline (no network), point a
config at it, and run a couple of optimizer steps through the real
``build_trainer`` path. Two variants:

* ``test_sft_text_end_to_end_cpu`` -- marked ``slow``; runs on CPU. Exercises the
  full text-SFT pipeline without a GPU.
* ``test_sft_text_end_to_end_gpu`` -- marked ``gpu``; same, forced onto CUDA in
  bf16. Skipped automatically when no GPU is present (see ``conftest.py``).

Run only the CPU suite:    pytest -m "not gpu"
Run the GPU suite:         pytest -m gpu
Include the slow CPU one:  pytest -m "slow and not gpu"
"""

import pytest

from glide.config.schema import GlideConfig, Modality, Task
from glide.trainers import build_trainer


def _build_tiny_model_dir(tmp_path, tokenizer):
    """Create + save a tiny LlamaForCausalLM and tokenizer; return the dir path."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(cfg)
    model_dir = tmp_path / "tiny_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return str(model_dir)


def _make_config(model_dir, train_path, output_dir, *, bf16=False) -> GlideConfig:
    config = GlideConfig()
    config.task = Task.SFT
    config.modality = Modality.TEXT
    config.model.name = model_dir
    config.model.attn_implementation = "eager"
    config.model.torch_dtype = "bfloat16" if bf16 else "float32"
    config.model.gradient_checkpointing = False
    config.data.train = train_path
    config.template.max_length = 64
    config.logging.report_to = ["none"]
    config.training = {
        "output_dir": output_dir,
        "max_steps": 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-3,
        "logging_steps": 1,
        "save_strategy": "no",
        "bf16": bf16,
        "report_to": "none",
        "dataloader_num_workers": 0,
    }
    return config


def _run(config) -> None:
    trainer = build_trainer(config)
    trainer.train()
    assert trainer.state.global_step >= 2


@pytest.mark.slow
def test_sft_text_end_to_end_cpu(tmp_path, fast_tokenizer, jsonl_file):
    records = [{"prompt": "hello", "response": "world"} for _ in range(8)]
    train = jsonl_file(records)
    model_dir = _build_tiny_model_dir(tmp_path, fast_tokenizer)
    config = _make_config(model_dir, train, str(tmp_path / "out"), bf16=False)
    config.training["use_cpu"] = True
    _run(config)


@pytest.mark.gpu
def test_sft_text_end_to_end_gpu(tmp_path, fast_tokenizer, jsonl_file):
    records = [{"prompt": "hello", "response": "world"} for _ in range(8)]
    train = jsonl_file(records)
    model_dir = _build_tiny_model_dir(tmp_path, fast_tokenizer)
    config = _make_config(model_dir, train, str(tmp_path / "out"), bf16=True)
    _run(config)
