"""Record -> chat-messages normalization and media-placeholder insertion.

A JSONL record can be expressed in several shapes; ``glide`` normalizes all of
them to a list of chat ``messages`` (the format the TRL trainers and HF
processors understand):

* **Conversational**: ``{"messages": [{"role": "user", "content": ...}, ...]}``
* **Prompt/response**: ``{"prompt": "...", "response": "..."}``
* **Multimodal**: any of the above plus an ``audio`` / ``image`` field; a media
  content block is prepended to the first user turn.

For RL (GRPO/GSPO) only the *prompt* side is needed; :func:`build_prompt_messages`
returns the conversation up to (but excluding) the final assistant turn.
"""

from typing import Any

from ..config.schema import DataConfig, Modality, TemplateConfig

__all__ = ["build_messages", "build_prompt_messages", "extract_media", "extract_reference"]


def _ensure_system(messages: list[dict], template: TemplateConfig) -> list[dict]:
    if template.system_prompt and not (messages and messages[0].get("role") == "system"):
        return [{"role": "system", "content": template.system_prompt}, *messages]
    return messages


def _media_block(modality: Modality, record: dict, data: DataConfig) -> dict | None:
    """Return a media content block (``{"type": "audio"|"image", ...}``) or None."""
    if modality is Modality.SPEECH and data.audio_field in record:
        return {"type": "audio", "audio": record[data.audio_field]}
    if modality is Modality.VISION and data.image_field in record:
        return {"type": "image", "image": record[data.image_field]}
    return None


def build_messages(
    record: dict[str, Any],
    data: DataConfig,
    template: TemplateConfig,
    modality: Modality = Modality.TEXT,
) -> list[dict[str, Any]]:
    """Normalize ``record`` into a full chat-message list (prompt + response).

    The returned messages use the OpenAI-style ``content`` which is either a plain
    string (text-only) or a list of typed blocks (multimodal). A media block, when
    present, is inserted at the front of the first user message.
    """
    if data.messages_field in record:
        messages = [dict(m) for m in record[data.messages_field]]
    else:
        prompt = record.get(data.prompt_field, "")
        messages = [{"role": "user", "content": prompt}]
        if data.response_field in record:
            messages.append({"role": "assistant", "content": record[data.response_field]})

    media = _media_block(modality, record, data)
    if media is not None:
        messages = _inject_media(messages, media)

    return _ensure_system(messages, template)


def _inject_media(messages: list[dict], media: dict) -> list[dict]:
    """Prepend a media block to the first user turn (promoting str content to list)."""
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                m["content"] = [media, {"type": "text", "text": content}]
            elif isinstance(content, list):
                m["content"] = [media, *content]
            break
    return out


def build_prompt_messages(
    record: dict[str, Any],
    data: DataConfig,
    template: TemplateConfig,
    modality: Modality = Modality.TEXT,
) -> list[dict[str, Any]]:
    """Like :func:`build_messages` but drops the trailing assistant turn.

    Used for RL prompts and validation-time generation.
    """
    messages = build_messages(record, data, template, modality)
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return messages


def extract_media(record: dict, data: DataConfig, modality: Modality):
    """Return the raw media reference (path or array) for ``record``, or ``None``."""
    if modality is Modality.SPEECH:
        return record.get(data.audio_field)
    if modality is Modality.VISION:
        return record.get(data.image_field)
    return None


def extract_reference(record: dict, data: DataConfig) -> str | None:
    """Return the reference text used by validation metrics, if available."""
    if data.reference_field in record:
        return record[data.reference_field]
    if data.response_field in record:
        resp = record[data.response_field]
        return resp if isinstance(resp, str) else None
    return None
