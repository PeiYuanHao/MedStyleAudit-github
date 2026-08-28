"""Stable manifest schema for the derived final-experiment repository."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from .checksums import sha256_file, verify_sha256


ALLOWED_TOP_LEVEL = {"README.md", "MANIFEST.json", "protocol", "p0", "checkpoints", "predictions", "audits", "aggregate", "figures", "logs"}
FORBIDDEN_PARTS = {"camelyon17", "wsi", "xml_annotations", "raw", "wilds"}


def validate_hf_path(path: str) -> str:
    normalized = str(PurePosixPath(path.replace("\\", "/"))).lstrip("/")
    parts = PurePosixPath(normalized).parts
    if not parts or parts[0] not in ALLOWED_TOP_LEVEL:
        raise ValueError(f"Artifact path is outside the frozen repository layout: {path}")
    lowered = {part.lower() for part in parts}
    if lowered & FORBIDDEN_PARTS:
        raise ValueError(f"Raw/source medical data paths may not be uploaded: {path}")
    if PurePosixPath(normalized).suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svs", ".xml"} and parts[:2] != ("figures", "final"):
        raise ValueError(f"Image/annotation files are only permitted for final figures, never medical source data: {path}")
    return normalized


@dataclass(frozen=True)
class ManifestEntry:
    local_path: str
    hf_path: str
    sha256: str
    bytes: int
    experiment: str | None = None
    backbone: str | None = None
    seed: int | None = None
    split: str | None = None
    git_commit: str | None = None
    protocol_hash: str | None = None

    @classmethod
    def from_file(cls, local_path: str | Path, hf_path: str, **metadata: object) -> "ManifestEntry":
        path = Path(local_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        filtered = {key: value for key, value in metadata.items() if key in cls.__dataclass_fields__}
        if filtered.get("seed") is not None:
            filtered["seed"] = int(filtered["seed"])
        return cls(str(path), validate_hf_path(hf_path), sha256_file(path), path.stat().st_size, **filtered)


class ArtifactManifest:
    version = 1

    def __init__(self, entries: Iterable[ManifestEntry] = ()) -> None:
        self.entries = list(entries)

    def add(self, entry: ManifestEntry) -> None:
        self.entries = [item for item in self.entries if item.hf_path != entry.hf_path]
        self.entries.append(entry)
        self.entries.sort(key=lambda item: item.hf_path)

    def to_dict(self) -> dict:
        return {"manifest_version": self.version, "artifact_count": len(self.entries), "artifacts": [asdict(item) for item in self.entries]}

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def read(cls, path: str | Path) -> "ArtifactManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(data.get("manifest_version", -1)) != cls.version:
            raise ValueError("Unsupported artifact manifest version")
        entries = [ManifestEntry(**item) for item in data.get("artifacts", [])]
        if len(entries) != int(data.get("artifact_count", len(entries))):
            raise ValueError("Manifest artifact_count does not match its entries")
        return cls(entries)

    def verify_local(self, root: str | Path | None = None) -> None:
        base = Path(root).resolve() if root else None
        for entry in self.entries:
            path = (base / Path(entry.hf_path)) if base else Path(entry.local_path)
            if not path.is_file():
                raise FileNotFoundError(f"Manifest artifact is missing: {path}")
            if path.stat().st_size != entry.bytes:
                raise ValueError(f"Size mismatch for {path}: expected {entry.bytes}, got {path.stat().st_size}")
            verify_sha256(path, entry.sha256)
