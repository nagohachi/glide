"""Regression test for #9: qwen_omni_aut audio tower forward contract."""

import torch


def test_omni_tower_forward_shapes(monkeypatch):
    import transformers
    from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as m

    from glide.models.encoders import _build_omni_tower

    cfg = m.Qwen2_5OmniAudioEncoderConfig(
        num_mel_bins=8, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=16,
        d_model=16, max_source_positions=512, n_window=4, output_dim=24,
    )
    tower = m.Qwen2_5OmniAudioEncoder(cfg).eval()
    # Avoid the network: the feature extractor is only used by the collator, not forward.
    monkeypatch.setattr(transformers.AutoFeatureExtractor, "from_pretrained",
                        staticmethod(lambda *a, **k: object()))

    enc = _build_omni_tower(tower, repo="dummy", out_dim=cfg.output_dim, freeze=True)
    assert enc.output_dim == 24  # config.output_dim, not d_model (16)

    b, mels, t = 2, 8, 48
    feats = torch.randn(b, mels, t)
    attn = torch.ones(b, t, dtype=torch.long)
    attn[1, 24:] = 0  # second utterance is half length
    with torch.no_grad():
        hidden, mask = enc.forward(feats, feature_attention_mask=attn)
    # Un-flattened back to padded (B, T, H) with H == output_dim.
    assert hidden.shape[0] == b and hidden.shape[-1] == 24
    assert mask.shape[0] == b
    # The shorter utterance yields fewer valid frames than the longer one.
    assert int(mask[1].sum()) < int(mask[0].sum())
