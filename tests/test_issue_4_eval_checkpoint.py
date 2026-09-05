"""Regression test for #4: `glide eval/test -c` checkpoint override needs `--` prefix."""


def test_checkpoint_override_prefix_parses():
    from glide.config.loader import parse_override_args

    # This is exactly the token _run_eval / _run_test now build for `-c <ckpt>`.
    parsed = parse_override_args(["--model.model_name_or_id=/path/to/ckpt"])
    assert parsed == {"model": {"model_name_or_id": "/path/to/ckpt"}}
