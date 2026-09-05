"""Regression test for #14: data.test resolved against the corpus root."""


def test_data_test_resolved_against_corpus_root(tmp_path):
    from glide.config.loader import load_config

    (tmp_path / "run.yaml").write_text(
        "model:\n  model_name_or_id: m\n  attn_implementation: sdpa\n"
        "data_roots:\n  csj: /abs/csj\n"
        "data:\n  corpus: csj\n  train: a/train.jsonl\n  eval: a/dev.jsonl\n"
        "  test: a/test.jsonl\n"
    )
    cfg = load_config(tmp_path / "run.yaml")
    assert cfg.data.test == "/abs/csj/a/test.jsonl"
