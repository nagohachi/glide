"""Tests for special-token registration (tokenizer side, no model needed)."""

from glide.config.schema import SpecialTokensConfig
from glide.models.special_tokens import apply_special_tokens


def test_adds_multimodal_tokens_and_resolves_ids(fast_tokenizer):
    cfg = SpecialTokensConfig(
        additional=["<audio>", "<audio_pad>", "<image>", "<image_pad>"],
        audio_token="<audio>",
        audio_pad_token="<audio_pad>",
        image_token="<image>",
        image_pad_token="<image_pad>",
        resize_embeddings=False,
    )
    before = len(fast_tokenizer)
    info = apply_special_tokens(fast_tokenizer, None, cfg)
    after = len(fast_tokenizer)

    assert after == before + 4
    assert info.num_added == 4
    assert info.audio_token_id == fast_tokenizer.convert_tokens_to_ids("<audio>")
    assert info.image_pad_token_id == fast_tokenizer.convert_tokens_to_ids("<image_pad>")
    # The markers are treated as atomic special tokens.
    assert fast_tokenizer.encode("<audio>", add_special_tokens=False) == [info.audio_token_id]


def test_idempotent_no_duplicate_adds(fast_tokenizer):
    cfg = SpecialTokensConfig(additional=["<audio>"], resize_embeddings=False)
    apply_special_tokens(fast_tokenizer, None, cfg)
    size = len(fast_tokenizer)
    info2 = apply_special_tokens(fast_tokenizer, None, cfg)
    assert info2.num_added == 0
    assert len(fast_tokenizer) == size
