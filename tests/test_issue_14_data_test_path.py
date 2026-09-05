"""Regression test for #14: data.test_jsonl_path resolved against the data root."""


def test_data_test_resolved_against_data_root(tmp_path):
    from glide.config.loader import load_config

    (tmp_path / "run.yaml").write_text(
        "model:\n  model_name_or_id: m\n  attn_implementation: sdpa\n"
        "data_roots:\n  csj: /abs/csj\n"
        "data:\n  root_key: csj\n  train_jsonl_path: a/train.jsonl\n  eval_jsonl_path: a/dev.jsonl\n"
        "  test_jsonl_path: a/test.jsonl\n"
    )
    cfg = load_config(tmp_path / "run.yaml")
    assert cfg.data.test_jsonl_path == "/abs/csj/a/test.jsonl"
