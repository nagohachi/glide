"""Special-token registration driven entirely by YAML.

This handles two things the spec calls out:

1. Adding arbitrary additional special tokens (e.g. ``<audio>`` ``<audio_pad>``
   ``<image>`` ``<image_pad>``) and standard tokens (bos/eos/pad).
2. Resizing the model's input/output embeddings to match, padded to a
   kernel-friendly multiple.

New token rows are initialized to the mean of the existing embeddings, which is a
better starting point than random vectors for fine-tuning.
"""

from ..config.schema import SpecialTokensConfig

__all__ = ["apply_special_tokens", "SpecialTokenInfo"]


class SpecialTokenInfo:
    """Resolved special-token ids, attached to the tokenizer/processor result."""

    def __init__(self):
        self.audio_token_id: int | None = None
        self.audio_pad_token_id: int | None = None
        self.image_token_id: int | None = None
        self.image_pad_token_id: int | None = None
        self.num_added: int = 0


def _get_tokenizer(processor):
    """Return the underlying tokenizer for either a processor or a tokenizer."""
    return getattr(processor, "tokenizer", processor)


def apply_special_tokens(processor, model, cfg: SpecialTokensConfig) -> SpecialTokenInfo:
    """Add special tokens from ``cfg`` to the tokenizer and resize ``model``.

    Args:
        processor: An HF processor or tokenizer.
        model: The model whose embeddings should be resized (may be ``None`` to
            only mutate the tokenizer, e.g. in unit tests).
        cfg: The :class:`SpecialTokensConfig`.

    Returns:
        A :class:`SpecialTokenInfo` with resolved token ids.
    """
    tokenizer = _get_tokenizer(processor)
    info = SpecialTokenInfo()

    # Standard tokens.
    standard = {}
    if cfg.bos_token:
        standard["bos_token"] = cfg.bos_token
    if cfg.eos_token:
        standard["eos_token"] = cfg.eos_token
    if cfg.pad_token:
        standard["pad_token"] = cfg.pad_token
    if standard:
        info.num_added += tokenizer.add_special_tokens(standard)

    # A missing pad token breaks batching; fall back to eos.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Collect additional + semantic multimodal markers (dedup, preserve order).
    additional: list[str] = list(cfg.additional)
    for tok in (cfg.audio_token, cfg.audio_pad_token, cfg.image_token, cfg.image_pad_token):
        if tok and tok not in additional:
            additional.append(tok)
    if additional:
        existing = set(tokenizer.get_vocab())
        new = [t for t in additional if t not in existing]
        if new:
            info.num_added += tokenizer.add_special_tokens(
                {"additional_special_tokens": new}
            )

    if cfg.resize_embeddings and model is not None and info.num_added > 0:
        _resize_with_mean_init(model, len(tokenizer), cfg.pad_to_multiple_of)

    # Resolve ids for the semantic markers.
    def _id(tok):
        return tokenizer.convert_tokens_to_ids(tok) if tok else None

    info.audio_token_id = _id(cfg.audio_token)
    info.audio_pad_token_id = _id(cfg.audio_pad_token)
    info.image_token_id = _id(cfg.image_token)
    info.image_pad_token_id = _id(cfg.image_pad_token)
    return info


def _resize_with_mean_init(model, new_size: int, pad_to_multiple_of: int | None) -> None:
    """Resize embeddings, initializing new rows to the mean of existing ones."""
    import torch

    old_embeddings = model.get_input_embeddings()
    old_num = old_embeddings.weight.size(0)
    model.resize_token_embeddings(new_size, pad_to_multiple_of=pad_to_multiple_of)

    if new_size > old_num:
        with torch.no_grad():
            inp = model.get_input_embeddings().weight
            mean = inp[:old_num].mean(dim=0)
            inp[old_num:] = mean
            out = model.get_output_embeddings()
            if out is not None and out.weight.size(0) >= new_size:
                out.weight[old_num:] = out.weight[:old_num].mean(dim=0)
