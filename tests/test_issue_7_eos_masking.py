"""Regression test for #7: EOS preserved when pad_token == eos_token."""

import torch

from glide.config.schema import DataConfig, TemplateConfig


def test_composed_collator_preserves_eos_when_pad_equals_eos(fast_tokenizer):
    from glide.data.composed_collator import ComposedSpeechCollator

    tok = fast_tokenizer
    tok.pad_token = tok.eos_token  # the Llama/Gemma fallback

    class _FE:
        def __call__(self, wavs, sampling_rate, return_tensors):
            return {"input_features": torch.zeros(len(wavs), 4, 6)}

    col = ComposedSpeechCollator(
        tokenizer=tok, feature_extractor=_FE(), input_kind="input_features",
        audio_token="<|resp|>", data=DataConfig(), template=TemplateConfig(),
        completion_only=False, train=False,
    )
    batch = col([{"prompt": "hi", "response": "hello world", "audio": [0.0] * 100}])
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    eos_id = tok.eos_token_id
    # The final real token is EOS and must remain supervised, even though
    # pad_token_id == eos_token_id would have masked it under id-based masking.
    last_real = max(i for i, m in enumerate(batch["attention_mask"][0].tolist()) if m == 1)
    assert ids[last_real] == eos_id
    assert labels[last_real] == eos_id
