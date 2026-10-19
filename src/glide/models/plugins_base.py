"""Base classes for pluggable multimodal components (audio encoders, projectors).

Subclass these in your project and register the *builder* with the appropriate
registry. glide calls the builders as

* audio encoder: ``builder(cfg: AudioEncoderConfig, sample_rate: int) -> AudioEncoder``
  (``cfg`` is ``speech.encoder``);
* projector: ``builder(cfg: ProjectorConfig, in_dim: int, out_dim: int) -> Projector``
  (``cfg`` is ``speech.projector``; ``in_dim`` is the encoder's ``output_dim``,
  ``out_dim`` the LLM hidden size).

For example::

    from glide.registry import audio_encoders
    from glide.models import AudioEncoder

    class MyWavLMEncoder(AudioEncoder):
        input_kind = "input_values"

        def __init__(self, cfg, sample_rate):
            super().__init__()
            ...
        def forward(self, input_values, attention_mask=None):
            ...
            return hidden_states, out_mask

    @audio_encoders.register("my_wavlm")
    def build_my_wavlm_encoder(cfg, sample_rate):
        return MyWavLMEncoder(cfg, sample_rate)

The base classes are deliberately thin -- they document the expected interface
and keep ``glide`` import-safe even when torch is the only heavy dependency
present.
"""

import torch.nn as nn

__all__ = ["AudioEncoder", "Projector"]


class AudioEncoder(nn.Module):
    """Encode raw/feature audio into a sequence of hidden states.

    Implementations must define :meth:`forward` returning
    ``(hidden_states, attention_mask)`` where ``hidden_states`` is
    ``(batch, frames, hidden)``.
    """

    output_dim: int = 0

    def forward(self, *args, **kwargs):  # pragma: no cover - interface stub
        raise NotImplementedError


class Projector(nn.Module):
    """Project encoder hidden states to the LLM embedding dimension.

    A common implementation is a small MLP (the ``proj1``/``proj2`` pattern in
    Qwen3-ASR). Implementations define :meth:`forward(hidden_states) -> tensor`.
    """

    def forward(self, *args, **kwargs):  # pragma: no cover - interface stub
        raise NotImplementedError
