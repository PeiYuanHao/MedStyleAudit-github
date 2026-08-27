"""Configuration loading and command-line override helpers."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` into a copied mapping."""
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML mapping, expand environment variables, and apply overrides."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {config_path}")
    parent: dict[str, Any] = {}
    extends = loaded.pop("extends", None)
    if extends:
        parent_path = Path(extends)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        if parent_path.resolve() == config_path.resolve():
            raise ValueError(f"Configuration cannot extend itself: {config_path}")
        parent = load_config(parent_path)
        parent.pop("_config_path", None)
    resolved = _expand(deep_update(parent, loaded))
    if overrides:
        resolved = deep_update(resolved, overrides)
    resolved["_config_path"] = str(config_path.resolve())
    return resolved


def require_keys(config: Mapping[str, Any], *keys: str) -> None:
    """Fail loudly if dotted configuration keys are absent."""
    for dotted in keys:
        value: Any = config
        for part in dotted.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise KeyError(f"Required configuration key is missing: {dotted}")
            value = value[part]
