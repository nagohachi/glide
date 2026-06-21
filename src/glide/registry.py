"""Lightweight registries and plugin discovery.

Every extensible component type (models, audio encoders, projectors, reward
functions, metrics, templates, data collators) has its own :class:`Registry`.
Built-ins register themselves at import time; user plugins register themselves
when their module is imported via :func:`load_plugins`.

Registering a component::

    from glide.registry import rewards

    @rewards.register("my_reward")
    def my_reward(completions, **kwargs):
        return [float(len(c)) for c in completions]

Resolving one::

    fn = rewards.get("my_reward")

Plugins are loaded from ``config.plugins`` (a list of dotted module paths or
``.py`` file paths). File paths are imported directly without touching
``sys.path``; bare module paths are imported with ``importlib`` and, only if that
fails, the current working directory (the project root from which ``glide`` was
invoked) is added to ``sys.path`` as a fallback.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")

__all__ = [
    "Registry",
    "load_plugins",
    "models",
    "audio_encoders",
    "projectors",
    "rewards",
    "metrics",
    "templates",
    "collators",
]


class Registry(Generic[T]):
    """A named collection of components of one kind."""

    def __init__(self, kind: str):
        self.kind = kind
        self._store: dict[str, T] = {}

    def register(self, name: str, obj: T | None = None, *, exist_ok: bool = False):
        """Register ``obj`` under ``name``.

        Usable as a decorator (``@reg.register("x")``) or directly
        (``reg.register("x", obj)``).
        """

        def _do(value: T) -> T:
            if name in self._store and not exist_ok:
                raise KeyError(
                    f"{self.kind!r} already has an entry named {name!r}. "
                    f"Pass exist_ok=True to override."
                )
            self._store[name] = value
            return value

        return _do if obj is None else _do(obj)

    def get(self, name: str) -> T:
        if name not in self._store:
            raise KeyError(
                f"No {self.kind} named {name!r}. Registered: {sorted(self._store)}"
            )
        return self._store[name]

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def names(self) -> list[str]:
        return sorted(self._store)


# Concrete registries shared across the package.
models: Registry[Callable] = Registry("model builder")
audio_encoders: Registry[Callable] = Registry("audio encoder")
projectors: Registry[Callable] = Registry("projector")
rewards: Registry[Callable] = Registry("reward function")
metrics: Registry[Callable] = Registry("metric")
templates: Registry[Callable] = Registry("template")
collators: Registry[Callable] = Registry("data collator")


def _import_file(path: Path) -> None:
    """Import a ``.py`` file as a standalone module (no ``sys.path`` mutation)."""
    mod_name = f"glide_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plugin file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)


def load_plugins(plugins: list[str]) -> None:
    """Import each plugin so its registrations take effect.

    Args:
        plugins: dotted module paths (``src.my_rewards``) or ``.py`` file paths
            (``src/my_rewards.py``), resolved relative to the current working dir.
    """
    for entry in plugins:
        if entry.endswith(".py") or os.sep in entry or "/" in entry:
            _import_file(Path(entry).resolve())
            continue
        try:
            importlib.import_module(entry)
        except ModuleNotFoundError:
            # Fallback: the project root (cwd) is typically not on sys.path for a
            # console script. Add it once and retry -- this is the "really needed"
            # case the contributor guide calls out.
            cwd = os.getcwd()
            if cwd not in sys.path:
                sys.path.insert(0, cwd)
            importlib.import_module(entry)
