"""Vision modality settings."""

from dataclasses import dataclass, field

__all__ = ["VisionConfig"]


@dataclass
class VisionConfig:
    """Vision-modality specific settings (vision + text input).

    The vision path is model-agnostic: it relies on the HF ``AutoProcessor`` and
    ``AutoModelForImageTextToText`` so any image-text-to-text model plugs in.
    """

    enabled: bool = False
    #: Longest edge / max pixels passed to the image processor (model dependent).
    max_pixels: int | None = None
    min_pixels: int | None = None
