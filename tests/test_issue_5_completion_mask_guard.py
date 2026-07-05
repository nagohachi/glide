"""Regression test for #5: completion-only masking without a usable response_template."""

import pytest

from glide.config.schema import DataConfig, Modality, TemplateConfig


def test_multimodal_collator_raises_without_response_template(fast_tokenizer):
    from glide.data.collator import MultimodalSFTCollator

    with pytest.raises(ValueError, match="train_on_completions_only"):
        MultimodalSFTCollator(
            processor=fast_tokenizer, data=DataConfig(),
            template=TemplateConfig(response_template=None),
            modality=Modality.SPEECH, completion_only=True,
        )
