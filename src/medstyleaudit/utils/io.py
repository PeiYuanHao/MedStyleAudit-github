"""Atomic, machine-readable experiment persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _atomic_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_json(data: Any, path: str | Path) -> Path:
    destination = Path(path)
    _atomic_text(destination, json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")
    return destination


def save_yaml(data: Any, path: str | Path) -> Path:
    destination = Path(path)
    _atomic_text(destination, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return destination


def save_table(frame: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Atomically persist CSV or Parquet based on the suffix."""
    destination = Path(path)
    ensure_dir(destination.parent)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if destination.suffix.lower() == ".parquet":
        frame.to_parquet(temporary, index=index)
    elif destination.suffix.lower() == ".csv":
        frame.to_csv(temporary, index=index)
    else:
        raise ValueError(f"Unsupported table format: {destination.suffix}")
    os.replace(temporary, destination)
    return destination


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported table format: {source.suffix}")
