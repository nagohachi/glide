"""Dataset builders read ``data.{split}_jsonl_path`` and the matching sample cap."""

from glide.config.schema import GlideConfig, Modality


def test_builders_resolve_split_to_jsonl_path(jsonl_file):
    from glide.data.build import build_multimodal_dataset, build_sft_text_dataset

    cfg = GlideConfig()
    cfg.modality = Modality.TEXT
    cfg.data.train_jsonl_path = jsonl_file(
        [{"prompt": f"q{i}", "response": f"a{i}"} for i in range(4)]
    )
    cfg.data.max_train_samples = 3

    assert len(build_sft_text_dataset(cfg, "train")) == 3
    assert len(build_multimodal_dataset(cfg, "train")) == 3
    # No eval path configured -> no dataset, not an AttributeError.
    assert build_sft_text_dataset(cfg, "eval") is None
