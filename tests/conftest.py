"""Shared pytest fixtures.

The fixtures here are deliberately *offline*: the unit suite must run without
network access or GPU. ``fast_tokenizer`` builds a byte-level BPE tokenizer from
scratch (byte-level guarantees any string is encodable) with a chat template and
the special tokens the collator/masking tests rely on.
"""

import json
import os

# Pin the test session to a single GPU *before* torch initializes CUDA. Otherwise
# HF Trainer wraps tiny test models in nn.DataParallel across every visible GPU,
# which can segfault. Override the device with GLIDE_TEST_GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GLIDE_TEST_GPU", "0"))

import pytest

# Chat template: every assistant turn is preceded by an atomic <|resp|> marker so
# completion-only masking can locate the response start by a single token id.
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{% if m['role'] == 'assistant' %}<|resp|>{% endif %}"
    "{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<|resp|>{% endif %}"
)
RESPONSE_TEMPLATE = "<|resp|>"


@pytest.fixture(scope="session")
def fast_tokenizer():
    """An offline byte-level BPE ``PreTrainedTokenizerFast`` with a chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = [
        "hello world", "transcribe the audio", "the quick brown fox",
        "good morning", "speech recognition is fun", "a b c d e f g",
    ] * 20
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=[])
    tok.train_from_iterator(corpus, trainer)

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
        chat_template=CHAT_TEMPLATE,
    )
    wrapped.add_special_tokens(
        {"additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<|resp|>"]}
    )
    return wrapped


@pytest.fixture
def jsonl_file(tmp_path):
    """Factory writing records to a temp JSONL file and returning its path."""

    def _make(records, name="data.jsonl"):
        path = tmp_path / name
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return str(path)

    return _make


def pytest_collection_modifyitems(config, items):
    """Skip ``gpu``-marked tests when CUDA is unavailable."""
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = False
    if has_gpu:
        return
    skip_gpu = pytest.mark.skip(reason="CUDA GPU not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
