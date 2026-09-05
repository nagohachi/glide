"""Regression test for #13: special-token resize on padded-vocab checkpoints."""

import torch
import torch.nn as nn

from glide.models.special_tokens import _resize_with_mean_init


class _FakeModel:
    """Minimal model exposing the resize API, mimicking a padded-vocab checkpoint."""

    def __init__(self, vocab, dim):
        self.inp = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab, bias=False)  # untied

    def get_input_embeddings(self):
        return self.inp

    def get_output_embeddings(self):
        return self.out

    def resize_token_embeddings(self, new, pad_to_multiple_of=None):
        if pad_to_multiple_of:
            new = ((new + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        for attr, layer, is_emb in (("inp", self.inp, True), ("out", self.out, False)):
            old = layer.weight
            new_layer = (nn.Embedding(new, old.shape[1]) if is_emb
                         else nn.Linear(old.shape[1], new, bias=False))
            with torch.no_grad():
                k = min(new, old.shape[0])
                new_layer.weight[:k] = old[:k]
            setattr(self, attr, new_layer)


def test_resize_with_mean_init_no_shrink_and_mean_init():
    # Qwen-style: embedding padded to 1000 rows but the real vocab was 990; add 1 token.
    m = _FakeModel(vocab=1000, dim=8)
    with torch.no_grad():
        m.inp.weight.normal_()
    real_vocab = 990
    _resize_with_mean_init(m, vocab_size=real_vocab + 1, num_added=1, pad_to_multiple_of=8)

    # Never shrank below the padded 1000 rows.
    assert m.inp.weight.shape[0] >= 1000
    # The new token row (id 990) is the mean of the prior real embeddings [0:990).
    expected = m.inp.weight[:real_vocab].mean(dim=0)
    assert torch.allclose(m.inp.weight[real_vocab], expected, atol=1e-5)
