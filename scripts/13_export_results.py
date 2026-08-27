"""Export GitHub-safe experiment records from external storage into the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_SUFFIXES = {".csv", ".json", ".yaml", ".yml", ".log", ".txt"}
EXCLUDED_PARTS = {"cache", "checkpoints", "data", "datasets", "models", ".git"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_results(source: Path, repository: Path, tag: str, max_file_mb: float, max_total_mb: float = 100.0, since_epoch: float | None = None) -> Path:
    if not source.is_dir():
        raise FileNotFoundError(f"Experiment output root does not exist: {source}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", tag):
        raise ValueError("Result tag may contain only letters, digits, dot, underscore, and hyphen")
    destination = (repository / "results" / tag).resolve()
    results_root = (repository / "results").resolve()
    if results_root not in destination.parents:
        raise ValueError("Destination escaped the repository results directory")
    if destination.exists():
        raise FileExistsError(f"Result destination already exists: {destination}")
    destination.mkdir(parents=True)
    limit = int(max_file_mb * 1024 * 1024)
    total_limit = int(max_total_mb * 1024 * 1024)
    total_copied = 0
    copied, skipped = [], []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if since_epoch is not None and path.stat().st_mtime < since_epoch:
            skipped.append({"path": str(relative), "reason": "older than this AutoDL job", "bytes": path.stat().st_size})
            continue
        if any(part.casefold() in EXCLUDED_PARTS for part in relative.parts) or path.suffix.casefold() not in ALLOWED_SUFFIXES:
            skipped.append({"path": str(relative), "reason": "excluded type or directory", "bytes": path.stat().st_size})
            continue
        size = path.stat().st_size
        if size > limit:
            skipped.append({"path": str(relative), "reason": f"larger than {max_file_mb:g} MiB GitHub export limit", "bytes": size})
            continue
        if total_copied + size > total_limit:
            skipped.append({"path": str(relative), "reason": f"would exceed {max_total_mb:g} MiB total GitHub export limit", "bytes": size})
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append({"path": str(relative), "bytes": size, "sha256": _sha256(target)})
        total_copied += size
    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source.resolve()),
        "max_file_mb": max_file_mb,
        "max_total_mb": max_total_mb,
        "total_copied_bytes": total_copied,
        "since_epoch": since_epoch,
        "copied": copied,
        "skipped": skipped,
        "note": "Datasets, caches, checkpoints, model weights, and oversized files remain on external server storage.",
    }
    (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(os.environ["MEDSTYLE_OUTPUT_ROOT"]) if "MEDSTYLE_OUTPUT_ROOT" in os.environ else None)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ"))
    parser.add_argument("--max-file-mb", type=float, default=20.0)
    parser.add_argument("--max-total-mb", type=float, default=100.0)
    parser.add_argument("--since-epoch", type=float, default=None)
    args = parser.parse_args()
    if args.source is None:
        parser.error("--source is required when MEDSTYLE_OUTPUT_ROOT is unset")
    print(export_results(args.source, args.repository, args.tag, args.max_file_mb, args.max_total_mb, args.since_epoch))


if __name__ == "__main__":
    main()
