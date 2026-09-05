"""Tests for the multimodal collator's completion-only label masking."""

from glide.config.schema import DataConfig, Modality, TemplateConfig
from glide.data.collator import MultimodalSFTCollator, find_subsequence


def test_find_subsequence_last_occurrence():
    assert find_subsequence([1, 2, 3, 2, 3, 4], [2, 3]) == 3
    assert find_subsequence([1, 2, 3], [9]) == -1
    assert find_subsequence([1, 2, 3], []) == -1


def test_completion_only_masking(fast_tokenizer):
    # Use the tokenizer itself as a stand-in "processor" (text-only path).
    collator = MultimodalSFTCollator(
        processor=fast_tokenizer,
        data=DataConfig(),
        template=TemplateConfig(response_template="<|resp|>"),
        modality=Modality.SPEECH,
        completion_only=True,
    )
    resp_id = fast_tokenizer.convert_tokens_to_ids("<|resp|>")
    assert collator.response_template_ids == [resp_id]

    batch = collator([{"prompt": "transcribe the audio", "response": "hello world"}])
    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    resp_pos = input_ids.index(resp_id)
    # Everything up to and including the <|resp|> marker is masked.
    assert all(x == -100 for x in labels[: resp_pos + 1])
    # At least one response token after the marker is supervised.
    assert any(x != -100 for x in labels[resp_pos + 1 :])


def test_rows_without_response_template_are_fully_masked(fast_tokenizer):
    collator = MultimodalSFTCollator(
        processor=fast_tokenizer,
        data=DataConfig(),
        template=TemplateConfig(response_template="<|resp|>"),
        modality=Modality.SPEECH,
        completion_only=True,
        response_template_ids=[999999],  # id that never appears
    )
    batch = collator([{"prompt": "hi", "response": "yo"}])
    assert all(x == -100 for x in batch["labels"][0].tolist())
