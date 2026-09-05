"""Tests for the registry and plugin discovery."""

import pytest

from glide.registry import Registry, load_plugins, rewards


def test_register_and_get():
    reg = Registry("thing")
    reg.register("a", 1)

    @reg.register("b")
    def b():
        return 2

    assert reg.get("a") == 1
    assert reg.get("b") is b
    assert set(reg.names()) == {"a", "b"}
    assert "a" in reg


def test_duplicate_rejected_unless_exist_ok():
    reg = Registry("thing")
    reg.register("a", 1)
    with pytest.raises(KeyError):
        reg.register("a", 2)
    reg.register("a", 2, exist_ok=True)
    assert reg.get("a") == 2


def test_get_missing_raises():
    reg = Registry("thing")
    with pytest.raises(KeyError, match="No thing named"):
        reg.get("missing")


def test_builtin_rewards_registered():
    for name in ("format", "length", "exact_match", "cer"):
        assert name in rewards


def test_load_plugin_from_file(tmp_path):
    plugin = tmp_path / "my_plugin.py"
    plugin.write_text(
        "from glide.registry import rewards\n"
        "@rewards.register('plugin_reward', exist_ok=True)\n"
        "def build():\n"
        "    return lambda prompts=None, completions=None, **k: [1.0]*len(completions)\n"
    )
    load_plugins([str(plugin)])
    assert "plugin_reward" in rewards
