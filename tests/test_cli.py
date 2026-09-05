"""Tests for the CLI parser and argument routing."""

import pytest

from glide.cli.main import _TRAIN_TASKS, _build_parser, main


def test_parser_has_all_commands():
    parser = _build_parser()
    # Parse each training command with a config + an override (override -> unknown).
    for cmd in _TRAIN_TASKS:
        args, overrides = parser.parse_known_args([cmd, "cfg.yaml", "--model.model_name_or_id", "x"])
        assert args.command == cmd
        assert args.config == "cfg.yaml"
        assert overrides == ["--model.model_name_or_id", "x"]


def test_eval_and_docs_commands():
    parser = _build_parser()
    args, _ = parser.parse_known_args(["eval", "cfg.yaml"])
    assert args.command == "eval"
    args, _ = parser.parse_known_args(["docs", "-o", "site"])
    assert args.command == "docs" and args.output == "site"


def test_version_exits():
    with pytest.raises(SystemExit):
        main(["--version"])
