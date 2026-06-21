"""Tests for record -> messages normalization and media injection."""

from glide.config.schema import DataConfig, Modality, TemplateConfig
from glide.data.template import (
    build_messages,
    build_prompt_messages,
    extract_media,
    extract_reference,
)

DATA = DataConfig()
TPL = TemplateConfig()


def test_prompt_response_to_messages():
    rec = {"prompt": "hi", "response": "hello"}
    msgs = build_messages(rec, DATA, TPL)
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_existing_messages_passthrough():
    rec = {"messages": [{"role": "user", "content": "x"}]}
    assert build_messages(rec, DATA, TPL)[0]["content"] == "x"


def test_system_prompt_prepended():
    tpl = TemplateConfig(system_prompt="be nice")
    msgs = build_messages({"prompt": "hi", "response": "yo"}, DATA, tpl)
    assert msgs[0] == {"role": "system", "content": "be nice"}


def test_audio_media_injected_into_first_user_turn():
    rec = {"audio": "/x.wav", "prompt": "transcribe", "response": "hello"}
    msgs = build_messages(rec, DATA, TPL, Modality.SPEECH)
    content = msgs[0]["content"]
    assert content[0] == {"type": "audio", "audio": "/x.wav"}
    assert content[1] == {"type": "text", "text": "transcribe"}


def test_image_media_injected():
    rec = {"image": "/x.jpg", "prompt": "describe", "response": "a cat"}
    msgs = build_messages(rec, DATA, TPL, Modality.VISION)
    assert msgs[0]["content"][0] == {"type": "image", "image": "/x.jpg"}


def test_prompt_messages_drops_assistant():
    rec = {"prompt": "hi", "response": "hello"}
    msgs = build_prompt_messages(rec, DATA, TPL)
    assert msgs == [{"role": "user", "content": "hi"}]


def test_extract_media_and_reference():
    rec = {"audio": "/a.wav", "reference": "gold"}
    assert extract_media(rec, DATA, Modality.SPEECH) == "/a.wav"
    assert extract_reference(rec, DATA) == "gold"
    assert extract_reference({"response": "r"}, DATA) == "r"
