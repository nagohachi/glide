"""The root :class:`GlideConfig` object assembled from every section."""

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .data import DataConfig, PackingConfig, TemplateConfig
from .enums import Modality, Task
from .evaluation import EvalConfig
from .model import ModelConfig, PeftConfigSpec, SpecialTokensConfig
from .rl import RLConfig
from .runtime import DistributedConfig, LoggingConfig
from .speech import SpeechConfig
from .vision import VisionConfig

__all__ = ["GlideConfig"]


@dataclass
class GlideConfig:
    """Root configuration object assembled from the merged YAML + CLI overrides."""

    task: Task = Task.SFT
    modality: Modality = Modality.TEXT
    seed: int = 42

    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PeftConfigSpec = field(default_factory=PeftConfigSpec)
    special_tokens: SpecialTokensConfig = field(default_factory=SpecialTokensConfig)
    data: DataConfig = field(default_factory=DataConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    #: Machine-specific name -> absolute data root map, selected by
    #: ``data.root_key``. Keep in a gitignored ``data_root.yaml`` pulled in via
    #: ``extends`` (see ``configs/data_root.example.yaml``).
    data_roots: dict[str, str] = field(default_factory=dict)

    #: Module paths (dotted) or file paths to import for plugin registration,
    #: e.g. ``["src.my_rewards", "src/my_encoder.py"]``.
    plugins: list[str] = field(default_factory=list)

    #: Free-form arguments forwarded to the TRL config object. ``output_dir`` here
    #: is versioned to ``output_dir/v{N}-{datetime}`` at run start.
    training: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain (YAML-serializable) dict of the whole config."""
        return _to_plain(dataclasses.asdict(self))


def _to_plain(obj: Any) -> Any:
    """Recursively convert enums to their values for serialization."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj
