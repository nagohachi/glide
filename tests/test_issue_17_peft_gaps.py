"""Regression test for #17: composed-path PEFT targets only the LLM."""

from glide.config.schema import GlideConfig, Modality


def test_peft_config_composed_targets_llm_only():
    from glide.trainers.sft import _peft_config

    cfg = GlideConfig()
    cfg.modality = Modality.SPEECH
    cfg.speech.encoder.name = "whisper"
    cfg.peft.enabled = True  # default target_modules = "all-linear"
    pc = _peft_config(cfg)
    # 'all-linear' is replaced by an llm.-anchored regex; the projector stays fully
    # trainable via modules_to_save (so the modality bridge can learn).
    assert isinstance(pc.target_modules, str) and pc.target_modules.startswith("llm")
    assert "projector" in pc.modules_to_save
