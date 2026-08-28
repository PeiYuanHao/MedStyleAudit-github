"""Hugging Face Dataset repository transport for derived artifacts only."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

from .manifest import ArtifactManifest, ManifestEntry, validate_hf_path


DEFAULT_REPO = "PeiyuanHao/MedStyleAudit-Experiments"


def repository_id(value: str | None = None) -> str:
    repo = value or os.environ.get("MEDSTYLE_HF_REPO", DEFAULT_REPO)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError(f"Invalid Hugging Face repository id: {repo}")
    return repo


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("Install the 'hf' extra (huggingface-hub) to manage artifacts") from error
    return HfApi(token=os.environ.get("HF_TOKEN"))


def create_repository(repo_id: str | None = None, *, private: bool = True) -> str:
    repo = repository_id(repo_id)
    _api().create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    return repo


def _remote_manifest(api, repo: str) -> ArtifactManifest:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, repo_type="dataset", filename="MANIFEST.json", token=os.environ.get("HF_TOKEN"))
    except Exception as error:
        if type(error).__name__ in {"EntryNotFoundError", "RemoteEntryNotFoundError", "RepositoryNotFoundError"}:
            return ArtifactManifest()
        raise
    return ArtifactManifest.read(path)


def upload_artifacts(
    files: Iterable[tuple[str | Path, str]],
    *,
    repo_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
    manifest_path: str | Path | None = None,
) -> ArtifactManifest:
    """Upload changed files and atomically publish the merged manifest last."""
    repo, api = repository_id(repo_id), _api()
    create_repository(repo, private=True)
    manifest = _remote_manifest(api, repo)
    existing = {entry.hf_path: entry for entry in manifest.entries}
    for local, remote in files:
        normalized = validate_hf_path(remote)
        parts = PurePosixPath(normalized).parts
        inferred: dict[str, object] = {"experiment": parts[1] if parts[0] in {"p0", "audits", "aggregate"} and len(parts) > 1 else parts[0]}
        inferred["backbone"] = next((part for part in parts if part in {"resnet50", "densenet121"}), None)
        seed_part = next((part for part in parts if re.fullmatch(r"seed_\d+", part)), None)
        inferred["seed"] = int(seed_part.split("_", 1)[1]) if seed_part else None
        inferred["split"] = next((part for part in parts if part in {"train", "id_val", "val", "test", "ood_val", "ood_test"}), None)
        inferred.update({key: value for key, value in dict(metadata or {}).items() if value is not None})
        entry = ManifestEntry.from_file(local, normalized, **inferred)
        old = existing.get(entry.hf_path)
        if old is None or old.sha256 != entry.sha256 or old.bytes != entry.bytes:
            api.upload_file(path_or_fileobj=entry.local_path, path_in_repo=entry.hf_path, repo_id=repo, repo_type="dataset", commit_message=f"Upload {entry.hf_path}")
        manifest.add(entry)
    if manifest_path:
        local_manifest = manifest.write(manifest_path)
        api.upload_file(path_or_fileobj=str(local_manifest), path_in_repo="MANIFEST.json", repo_id=repo, repo_type="dataset", commit_message="Update artifact manifest")
    else:
        with tempfile.TemporaryDirectory() as directory:
            local_manifest = manifest.write(Path(directory) / "MANIFEST.json")
            api.upload_file(path_or_fileobj=str(local_manifest), path_in_repo="MANIFEST.json", repo_id=repo, repo_type="dataset", commit_message="Update artifact manifest")
    return manifest


def directory_files(root: str | Path, hf_prefix: str = "") -> list[tuple[Path, str]]:
    source = Path(root).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    pairs = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative_parts = path.relative_to(source).parts
        if path.name in {"MANIFEST.json", "last.ckpt"} or any(part.startswith(".") for part in relative_parts) or relative_parts[:2] == ("p0", "matching_final_test"):
            continue
        relative = path.relative_to(source).as_posix()
        if relative == "final_run_state.json":
            relative = "logs/final_suite/final_run_state.json"
        remote = str(PurePosixPath(hf_prefix) / PurePosixPath(relative))
        pairs.append((path, validate_hf_path(remote)))
    return pairs


def download_manifest_artifacts(destination: str | Path, *, repo_id: str | None = None) -> ArtifactManifest:
    repo, api = repository_id(repo_id), _api()
    manifest = _remote_manifest(api, repo)
    root = Path(destination)
    from huggingface_hub import hf_hub_download
    for entry in manifest.entries:
        hf_hub_download(repo_id=repo, repo_type="dataset", filename=entry.hf_path, local_dir=str(root), token=os.environ.get("HF_TOKEN"))
    manifest.write(root / "MANIFEST.json")
    manifest.verify_local(root)
    return manifest


def verify_remote_artifacts(*, repo_id: str | None = None) -> ArtifactManifest:
    with tempfile.TemporaryDirectory() as directory:
        return download_manifest_artifacts(directory, repo_id=repo_id)
