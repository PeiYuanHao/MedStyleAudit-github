"""Shared command-line behavior for experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path


def common_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-final-test", action="store_true")
    return parser


def enforce_final_test_guard(split: str, allowed: bool) -> None:
    if split in {"test", "ood_test"} and not allowed:
        raise PermissionError("Final hospital-2 evaluation requires --allow-final-test")
    if split in {"test", "ood_test"}:
        from medstyleaudit.protocol import authorize_final_test
        from medstyleaudit.utils.paths import storage_root
        authorize_final_test(storage_root("MEDSTYLE_OUTPUT_ROOT"))
