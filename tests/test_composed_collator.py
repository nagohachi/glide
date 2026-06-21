"""Tests for ComposedSpeechCollator message/thinking construction (no model needed)."""

from glide.config.schema import DataConfig, TemplateConfig
from glide.data.composed_collator import ComposedSpeechCollator


def _collator(enable_thinking):
    return ComposedSpeechCollator(
        tokenizer=None, feature_extractor=None, input_kind="input_features",
        audio_token="<audio>",
        data=DataConfig(response_field="text", reasoning_field="think"),
        template=TemplateConfig(enable_thinking=enable_thinking, system_prompt=""),
    )


def test_user_turn_carries_audio_marker():
    msgs = _collator(True)._messages({"text": "ans"}, with_target=False)
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"].startswith("<audio>")


def test_thinking_mode_adds_reasoning_content():
    msgs = _collator(True)._messages({"text": "ans", "think": "because"}, with_target=True)
    asst = msgs[-1]
    assert asst["role"] == "assistant" and asst["content"] == "ans"
    assert asst.get("reasoning_content") == "because"  # -> <think>because</think>ans


def test_no_think_mode_omits_reasoning_content():
    msgs = _collator(False)._messages({"text": "ans", "think": "because"}, with_target=True)
    assert "reasoning_content" not in msgs[-1]


def test_thinking_without_reasoning_field_is_plain():
    msgs = _collator(True)._messages({"text": "ans"}, with_target=True)  # no 'think' key
    assert "reasoning_content" not in msgs[-1]


def test_ct_kwargs_passes_enable_thinking_only_when_set():
    assert _collator(True)._ct_kwargs() == {"enable_thinking": True}
    assert _collator(False)._ct_kwargs() == {"enable_thinking": False}
    assert _collator(None)._ct_kwargs() == {}
