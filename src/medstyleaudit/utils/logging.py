"""Console and per-run file logging."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(output_dir: str | Path, experiment: str, run_id: str) -> logging.Logger:
    """Create an isolated console/file logger with experiment context."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"medstyleaudit.{experiment}.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        f"%(asctime)s | %(levelname)s | {experiment} | {run_id} | %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(directory / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
