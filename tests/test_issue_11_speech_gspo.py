"""Regression test for #11: speech GSPO weight-sync name mapping.

Covers the part of the bundle that is unit-testable offline: the trained-policy ->
vLLM-server param name map must NOT drop the trainable projector, and must push
lm_head for untied LLMs (skip it only when embeddings are tied).
"""

from glide.trainers.rl_speech import _glide_to_thinker


def test_glide_to_thinker_maps_projector_and_untied_lm_head():
    # Encoder + LLM body map to the server's thinker.* names.
    assert _glide_to_thinker("encoder.model.layers.0.q_proj.weight") == \
        "thinker.audio_tower.layers.0.q_proj.weight"
    assert _glide_to_thinker("llm.model.layers.0.mlp.up_proj.weight") == \
        "thinker.model.layers.0.mlp.up_proj.weight"
    # The trainable projector must be synced (was silently dropped -> None before).
    assert _glide_to_thinker("projector.net.0.weight") == "projector.net.0.weight"
    # lm_head: skipped only for tied embeddings; pushed for untied models.
    assert _glide_to_thinker("llm.lm_head.weight", tie_word_embeddings=True) is None
    assert _glide_to_thinker("llm.lm_head.weight", tie_word_embeddings=False) == \
        "thinker.lm_head.weight"
