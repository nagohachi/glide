"""Tests for built-in reward functions."""

from glide.plugins.rewards import (
    build_cer_reward,
    build_exact_match_reward,
    build_format_reward,
    build_length_reward,
    completion_text,
)


def test_completion_text_str_and_messages():
    assert completion_text("hi") == "hi"
    assert completion_text([{"role": "assistant", "content": "hello"}]) == "hello"
    # multimodal content blocks
    msg = [{"role": "assistant", "content": [{"type": "text", "text": "hey"}]}]
    assert completion_text(msg) == "hey"


def test_format_reward_matches_think_block():
    fn = build_format_reward()
    out = fn(completions=["<think>reasoning</think> answer", "no tags here"])
    assert out == [1.0, 0.0]


def test_length_reward_peaks_near_target():
    fn = build_length_reward(target=10, scale=0.1)
    near = fn(completions=["x" * 10])[0]
    far = fn(completions=["x" * 100])[0]
    assert near == 1.0 and far < near


def test_exact_match_reward_uses_reference_column():
    fn = build_exact_match_reward(reference_key="reference")
    out = fn(completions=["hello world", "wrong"], reference=["Hello, world!", "right"])
    assert out == [1.0, 0.0]  # normalization makes #0 match


def test_cer_reward_rewards_low_error():
    fn = build_cer_reward(reference_key="reference")
    out = fn(completions=["hello", "xxxxx"], reference=["hello", "hello"])
    assert out[0] == 1.0
    assert out[1] < out[0]


def test_cer_reward_without_reference_is_zero():
    fn = build_cer_reward()
    assert fn(completions=["hello"]) == [0.0]
