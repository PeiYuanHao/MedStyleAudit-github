"""Server-first storage paths kept outside the Git checkout."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def storage_root(variable: str) -> Path:
    """Return a required absolute persistent-storage root."""
    value = os.environ.get(variable)
    if not value:
        raise EnvironmentError(
            f"{variable} is not set. On the server, export {variable}="
            + ("/workspace/datasets" if variable == "MEDSTYLE_DATA_ROOT" else "/workspace/experiments/medstyleaudit")
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{variable} must be an absolute path, got: {path}")
    return path


def experiment_path(relative: str | Path) -> Path:
    return storage_root("MEDSTYLE_OUTPUT_ROOT") / relative


def configured_output(config: Mapping[str, Any], fallback_relative: str | Path) -> Path:
    value = str(config.get("output_dir", ""))
    if not value:
        return experiment_path(fallback_relative)
    if "${MEDSTYLE_OUTPUT_ROOT}" in value:
        return experiment_path(value.replace("${MEDSTYLE_OUTPUT_ROOT}", "").lstrip("/\\"))
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Experiment output must be outside the repository and absolute: {path}")
    return path
