"""Loading and merging of YAML configs, CLI overrides and experiment versioning.

Resolution order (lowest to highest precedence):

1. Dataclass defaults in :mod:`glide.config.schema`.
2. ``base.yaml`` (and any chain referenced via the ``extends:`` key).
3. The run-specific YAML passed on the command line.
4. ``--dotted.key value`` overrides given on the command line.

``extends`` may be a single path or a list; paths are resolved relative to the
file that declares them. Cycles are detected and rejected.
"""

import dataclasses
import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

from .schema import GlideConfig

__all__ = [
    "load_config",
    "deep_merge",
    "apply_overrides",
    "dict_to_dataclass",
    "version_output_dir",
    "build_training_args",
]


# --------------------------------------------------------------------------- #
# YAML loading with `extends`
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping at the top level.")
    return data


def _resolve_extends(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load ``path``, recursively merging any ``extends:`` parents underneath it."""
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        chain = " -> ".join(str(p) for p in _seen) + f" -> {path}"
        raise ValueError(f"Cyclic `extends` detected: {chain}")
    _seen = _seen | {path}

    raw = _read_yaml(path)
    parents = raw.pop("extends", None)
    if parents is None:
        return raw
    if isinstance(parents, str):
        parents = [parents]

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = (path.parent / parent).resolve()
        merged = deep_merge(merged, _resolve_extends(parent_path, _seen))
    return deep_merge(merged, raw)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins).

    Nested dicts are merged key-by-key; everything else (including lists) is
    replaced wholesale by ``override``.
    """
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


# --------------------------------------------------------------------------- #
# CLI overrides: --a.b.c value
# --------------------------------------------------------------------------- #
def _coerce_scalar(text: str) -> Any:
    """Best-effort coercion of a CLI string into a python scalar/JSON value."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    # JSON for lists/dicts/quoted strings/numbers.
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return text


def parse_override_args(tokens: list[str]) -> dict[str, Any]:
    """Parse ``--dotted.key value`` (or ``--dotted.key=value``) tokens.

    Returns a *nested* dict suitable for :func:`deep_merge`. Boolean flags given
    without a value (``--peft.enabled``) become ``True``.
    """
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected CLI token {tok!r}; expected --key.")
        key = tok[2:]
        if "=" in key:
            key, _, value = key.partition("=")
            val: Any = _coerce_scalar(value)
            i += 1
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            val = _coerce_scalar(tokens[i + 1])
            i += 2
        else:
            val = True  # bare flag
            i += 1
        _set_nested(overrides, key.split("."), val)
    return overrides


def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
        if not isinstance(d, dict):
            raise ValueError(f"Override path conflict at {k!r}.")
    d[keys[-1]] = value


def apply_overrides(config_dict: dict[str, Any], tokens: list[str]) -> dict[str, Any]:
    """Merge parsed CLI ``tokens`` on top of ``config_dict``."""
    return deep_merge(config_dict, parse_override_args(tokens))


# --------------------------------------------------------------------------- #
# dict -> dataclass hydration
# --------------------------------------------------------------------------- #
def dict_to_dataclass(cls: type, data: Any) -> Any:
    """Recursively build dataclass ``cls`` from ``data`` (a dict).

    Unknown keys raise a ``ValueError`` so typos surface early. Enum fields accept
    their string value. ``dict``-typed fields (notably ``training`` and
    ``extra_kwargs``) are passed through untouched.
    """
    if data is None:
        return cls() if dataclasses.is_dataclass(cls) else None
    if not dataclasses.is_dataclass(cls):
        return data

    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(fields)}"
        )

    # Resolve real field types (handles forward refs / string annotations) before
    # hydrating, rather than reading the possibly-stringized ``field.type``.
    import typing

    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name in fields:
        if name not in data:
            continue
        kwargs[name] = _hydrate_value(hints[name], data[name])
    return cls(**kwargs)


def _hydrate_value(ftype: Any, value: Any) -> Any:
    origin = get_origin(ftype)
    # An empty YAML section (`training:` with every key commented out) parses to None;
    # hydrate to the container default instead of leaving None to crash a later
    # `.setdefault(...)`. Optional scalars keep None (their intended value).
    if value is None:
        if dataclasses.is_dataclass(ftype):
            return ftype()
        if origin is dict:
            return {}
        if origin in (list, tuple):
            return []
        return None
    # Enum field.
    if isinstance(ftype, type) and issubclass(ftype, _enum_base()):
        return ftype(value)
    # Nested dataclass.
    if dataclasses.is_dataclass(ftype):
        return dict_to_dataclass(ftype, value)
    # list[X] / tuple[X].
    if origin in (list, tuple):
        # A scalar CLI override to a list field (`--eval.metrics cer`) arrives as the bare
        # scalar; wrap it so consumers that iterate don't walk the string char-by-char
        # (`build_metric_fn` -> KeyError: No metric named 'c').
        if not isinstance(value, (list, tuple)):
            value = [value]
        (item_type,) = get_args(ftype) or (Any,)
        if dataclasses.is_dataclass(item_type):
            return [dict_to_dataclass(item_type, v) for v in value]
    return value


def _enum_base():
    from enum import Enum

    return Enum


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_config(
    yaml_path: str | os.PathLike | None,
    overrides: list[str] | None = None,
    *,
    task: str | None = None,
) -> GlideConfig:
    """Load a fully-resolved :class:`GlideConfig`.

    Args:
        yaml_path: Path to the run YAML (may declare ``extends:``). May be ``None``
            to build a config purely from defaults + overrides.
        overrides: Raw ``--dotted.key value`` CLI tokens.
        task: Optional task name from the CLI subcommand (``sft``/``grpo``/...),
            which takes precedence over any ``task:`` in the YAML.
    """
    merged: dict[str, Any] = {}
    if yaml_path is not None:
        merged = _resolve_extends(Path(yaml_path))
    if overrides:
        merged = apply_overrides(merged, overrides)
    if task is not None:
        merged["task"] = task

    config = dict_to_dataclass(GlideConfig, merged)
    _resolve_data_paths(config)
    return config


def _resolve_data_paths(config: GlideConfig) -> None:
    """Prepend the corpus root to relative ``train``/``eval`` paths (in place).

    The root is ``data_roots[data.corpus]`` when ``data.corpus`` is set, else the
    explicit ``data.root``.
    """
    if config.data.corpus:
        if config.data.corpus not in config.data_roots:
            raise ValueError(
                f"data.corpus={config.data.corpus!r} not found in data_roots "
                f"{sorted(config.data_roots)}; define it in your data_root.yaml."
            )
        root = config.data_roots[config.data.corpus]
    else:
        root = config.data.root
    if not root:
        return

    def _join(p):
        if p is None:
            return None
        if isinstance(p, list):
            return [_join(x) for x in p]
        return p if os.path.isabs(p) else os.path.join(root, p)

    config.data.train = _join(config.data.train)
    config.data.eval = _join(config.data.eval)


# --------------------------------------------------------------------------- #
# Experiment versioning
# --------------------------------------------------------------------------- #
_VERSION_RE = re.compile(r"^v(\d+)-")


def version_output_dir(base_dir: str | os.PathLike, *, now: _dt.datetime | None = None) -> str:
    """Return ``base_dir/v{N}-{YYYYmmdd-HHMMSS}`` with an auto-incremented ``N``.

    ``N`` is one greater than the highest existing ``v{N}-*`` directory under
    ``base_dir`` (starting at 0). The directory is *not* created here; the trainer
    creates it. ``now`` is injectable for deterministic tests.
    """
    base = Path(base_dir)
    next_n = 0
    if base.is_dir():
        existing = []
        for child in base.iterdir():
            if child.is_dir():
                m = _VERSION_RE.match(child.name)
                if m:
                    existing.append(int(m.group(1)))
        if existing:
            next_n = max(existing) + 1
    if now is not None:
        stamp = now.strftime("%Y%m%d-%H%M%S")
    else:
        # Under DDP the launcher exports GLIDE_OUTPUT_STAMP so all ranks agree.
        stamp = os.environ.get("GLIDE_OUTPUT_STAMP") or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(base / f"v{next_n}-{stamp}")


# --------------------------------------------------------------------------- #
# TRL config construction
# --------------------------------------------------------------------------- #
def build_training_args(config: GlideConfig, trl_config_cls: type, *, now=None):
    """Instantiate a TRL config object (e.g. ``SFTConfig``) from ``config.training``.

    Only keys recognised by ``trl_config_cls`` (its dataclass fields) are passed;
    unknown keys raise ``ValueError`` so mistakes are caught. ``output_dir`` is
    versioned via :func:`version_output_dir` unless ``training.no_version`` is set.
    """
    raw = dict(config.training)
    no_version = bool(raw.pop("no_version", False))

    output_dir = raw.get("output_dir", "outputs")
    if not no_version:
        output_dir = version_output_dir(output_dir, now=now)
    raw["output_dir"] = output_dir

    # Wire logging / seed / gradient checkpointing from the structured config
    # unless explicitly set under `training`.
    report_to = [r for r in config.logging.report_to if r != "none"]
    raw.setdefault("report_to", report_to or "none")
    raw.setdefault("seed", config.seed)
    # Wire data.num_workers -> the dataloader worker count (it was otherwise never read;
    # actual workers came only from training.dataloader_num_workers). Explicit
    # training.dataloader_num_workers still wins.
    if "dataloader_num_workers" in {f.name for f in dataclasses.fields(trl_config_cls)}:
        raw.setdefault("dataloader_num_workers", config.data.num_workers)
    if config.model.gradient_checkpointing:
        raw.setdefault("gradient_checkpointing", True)
    # Non-reentrant checkpointing is required for gradient checkpointing under DDP
    # (reentrant marks tied/shared params ready twice -> "mark a variable ready
    # only once"). Default it on whenever checkpointing is enabled.
    if raw.get("gradient_checkpointing"):
        raw.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})
    if config.logging.run_name:
        raw.setdefault("run_name", config.logging.run_name)

    # bf16 auto-gating: transformers 5 resolves the default bf16=None by auto-enabling
    # bf16, which then fails TrainingArguments validation on a CPU-only host ("setup
    # doesn't support bf16/gpu"). Pin it to what the hardware actually supports unless the
    # user set bf16/fp16 explicitly.
    if "bf16" not in raw and "fp16" not in raw and "bf16" in {f.name for f in dataclasses.fields(trl_config_cls)}:
        try:
            import torch

            raw["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        except Exception:
            raw["bf16"] = False

    valid = {f.name for f in dataclasses.fields(trl_config_cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"Unknown `training` key(s) for {trl_config_cls.__name__}: "
            f"{sorted(unknown)}."
        )
    return trl_config_cls(**raw)
