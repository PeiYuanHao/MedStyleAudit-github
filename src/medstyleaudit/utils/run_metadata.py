"""Traceable run directory creation and environment capture."""

from __future__ import annotations

import platform
import socket
import subprocess
import sys
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .io import ensure_dir, save_json, save_yaml
from .logging import configure_logging


def _command(*args: str) -> str | None:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info() -> dict[str, Any]:
    """Collect software and hardware metadata without requiring PyTorch."""
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "git_commit": _command("git", "rev-parse", "HEAD"),
        "git_dirty": bool(_command("git", "status", "--porcelain")),
    }
    try:
        import torch

        info.update(
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
    except ImportError:
        info["torch_version"] = None
    return info


@dataclass
class RunContext:
    output_dir: Path
    run_id: str
    logger: Any
    info: dict[str, Any]

    def complete(self, status: str = "completed", **extra: Any) -> None:
        self.info.update(status=status, finished_at=datetime.now(timezone.utc).isoformat(), **extra)
        save_json(self.info, self.output_dir / "run_info.json")


def start_run(
    experiment: str,
    config: Mapping[str, Any],
    output_dir: str | Path,
    seed: int,
    *,
    overwrite: bool = False,
) -> RunContext:
    """Initialize a non-ambiguous run directory and its required records."""
    directory = Path(output_dir)
    if (directory / "run_info.json").exists() and not overwrite:
        raise FileExistsError(f"Run already exists; pass --overwrite to replace it: {directory}")
    ensure_dir(directory)
    run_id = f"{experiment}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{seed}"
    logger = configure_logging(directory, experiment, run_id)
    info = {
        "experiment": experiment,
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "config_file": config.get("_config_path"),
        "dataset_version": config.get("data", {}).get("version"),
        "checkpoint_path": config.get("checkpoint"),
        "output_path": str(directory.resolve()),
        **environment_info(),
    }
    output_root = os.environ.get("MEDSTYLE_OUTPUT_ROOT")
    if output_root:
        hash_path = Path(output_root) / "protocol" / "protocol_sha256.txt"
        if hash_path.is_file():
            digest = hash_path.read_text(encoding="utf-8").strip()
            info["protocol_hash"] = digest
            info["protocol_sha256"] = digest
    save_json(info, directory / "run_info.json")
    save_yaml(dict(config), directory / "config_resolved.yaml")
    logger.info("run initialized")
    return RunContext(directory, run_id, logger, info)
