"""Regression test for #6: composed collator does not add a second BOS."""

import torch

from glide.config.schema import DataConfig, TemplateConfig


def test_composed_collator_no_double_special_tokens(fast_tokenizer):
    from glide.data.composed_collator import ComposedSpeechCollator

    tok = fast_tokenizer

    class _FE:
        def __call__(self, wavs, sampling_rate, return_tensors):
            return {"input_features": torch.zeros(len(wavs), 4, 6)}

    col = ComposedSpeechCollator(
        tokenizer=tok, feature_extractor=_FE(), input_kind="input_features",
        audio_token="<|resp|>", data=DataConfig(), template=TemplateConfig(),
        completion_only=True, train=False,
    )
    rec = {"prompt": "hi", "response": "hello world", "audio": [0.0] * 100}
    full_text = col._full_text(rec)
    # Training ids must equal the chat-template render tokenized with NO extra special
    # tokens -- i.e. the collator did not prepend an auto-BOS on top of the template's.
    expected = tok(full_text, add_special_tokens=False)["input_ids"]
    batch = col([rec])
    assert batch["input_ids"][0].tolist() == expected
