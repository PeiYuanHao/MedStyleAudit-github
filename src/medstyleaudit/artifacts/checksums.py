"""Streaming checksums used by local and Hugging Face artifact verification."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
