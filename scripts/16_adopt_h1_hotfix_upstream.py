"""Explicitly adopt approved Hospital-1 upstream artifacts after the audit-only hotfix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medstyleaudit.artifacts.checksums import sha256_file
from medstyleaudit.protocol import protocol_sha256
from medstyleaudit.utils.config import load_config

SOURCE_COMMIT = "ea2f91f6821e4bbd2772393c786d3dc109ae769e"
PROTOCOL_HASH = "ab44aaa68bdd7a9e871f3edb1e9a6b959a61068c2b18786588d37e44da2d2b04"
REASON = "post-training audit-only hotfix"
ADOPTED_STAGES = (
    "descriptors",
    "matching",
    "train_resnet50_0011",
    "train_resnet50_0042",
    "train_resnet50_0101",
)
TRAINING_SEEDS = {
    "train_resnet50_0011": 11,
    "train_resnet50_0042": 42,
    "train_resnet50_0101": 101,
}
ALLOWED_DIFF_PATHS = {
    "src/medstyleaudit/audit/inference.py",
    "src/medstyleaudit/artifacts/manifest.py",
    "tests/test_batched_inference.py",
    "tests/test_artifact_manifest.py",
    "scripts/16_adopt_h1_hotfix_upstream.py",
    "tests/test_h1_hotfix_adoption.py",
}
REQUIRED_HOTFIX_PATHS = {
    "src/medstyleaudit/audit/inference.py",
    "src/medstyleaudit/artifacts/manifest.py",
}
LEDGER_RELATIVE_PATH = Path("protocol/h1_hotfix_upstream_adoption.json")
ARCHIVE_RELATIVE_PATH = Path("protocol/h1_hotfix_upstream_original_markers")


class AdoptionError(RuntimeError):
    """Raised when compatibility adoption cannot be proven safe."""


def _git(repository: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=check,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AdoptionError(f"Git validation command failed: git {' '.join(arguments)}") from error


def validate_diff_paths(diff_paths: list[str]) -> None:
    unexpected = sorted(set(diff_paths) - ALLOWED_DIFF_PATHS)
    if unexpected:
        raise AdoptionError(f"Unapproved source-to-target Git changes: {unexpected}")
    missing = sorted(REQUIRED_HOTFIX_PATHS - set(diff_paths))
    if missing:
        raise AdoptionError(f"Required approved audit hotfix changes are missing: {missing}")


def inspect_repository(repository: Path) -> tuple[str, list[str]]:
    """Return current commit and approved diff paths after strict Git validation."""
    if _git(repository, "status", "--porcelain").stdout.strip():
        raise AdoptionError("Compatibility adoption requires a clean Git working tree")
    resolved_source = _git(repository, "rev-parse", f"{SOURCE_COMMIT}^{{commit}}").stdout.strip()
    if resolved_source != SOURCE_COMMIT:
        raise AdoptionError(f"Source production commit does not resolve exactly to {SOURCE_COMMIT}")
    target_commit = _git(repository, "rev-parse", "HEAD").stdout.strip()
    ancestry = _git(repository, "merge-base", "--is-ancestor", SOURCE_COMMIT, target_commit, check=False)
    if ancestry.returncode != 0:
        raise AdoptionError("Current HEAD does not descend from the approved production commit")
    diff_paths = sorted(
        path
        for path in _git(repository, "diff", "--name-only", f"{SOURCE_COMMIT}..{target_commit}").stdout.splitlines()
        if path
    )
    validate_diff_paths(diff_paths)
    return target_commit, diff_paths


def verify_frozen_protocol(repository: Path, output_root: Path) -> None:
    tracked_hash_path = repository / "configs/final/protocol_sha256.txt"
    output_hash_path = output_root / "protocol/protocol_sha256.txt"
    for path in (tracked_hash_path, output_hash_path):
        if not path.is_file() or path.read_text(encoding="utf-8").strip() != PROTOCOL_HASH:
            raise AdoptionError(f"Frozen protocol hash does not match the approved production hash: {path}")
    actual = protocol_sha256(repository / "configs/final/FINAL_PROTOCOL.yaml")
    if actual != PROTOCOL_HASH:
        raise AdoptionError(
            f"Current FINAL_PROTOCOL.yaml hashes to {actual}, expected {PROTOCOL_HASH}"
        )


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise AdoptionError(f"Missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdoptionError(f"Invalid {description}: {path} ({error})") from error
    if not isinstance(value, dict):
        raise AdoptionError(f"Invalid {description}: expected a JSON object at {path}")
    return value


def _relative_output_path(path: Path, output_root: Path) -> str:
    try:
        return path.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError as error:
        raise AdoptionError(f"Recorded artifact is outside MEDSTYLE_OUTPUT_ROOT: {path}") from error


def _expected_outputs(stage: str) -> set[str]:
    if stage == "descriptors":
        return {
            "p0/descriptors/descriptors.parquet",
            "p0/descriptors/mask_stability.parquet",
        }
    if stage == "matching":
        return {
            "p0/matching/triplets.parquet",
            "p0/matching/feature_balance.csv",
            "p0/matching/donor_reuse.csv",
            "p0/matching/slide_reuse.csv",
        }
    seed = TRAINING_SEEDS[stage]
    return {
        f"checkpoints/resnet50/seed_{seed:04d}/best.ckpt",
        f"checkpoints/resnet50/seed_{seed:04d}/metrics.csv",
        f"predictions/resnet50/seed_{seed:04d}/id_val.parquet",
        f"predictions/resnet50/seed_{seed:04d}/ood_val.parquet",
    }


def _artifact_record(path: Path, output_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AdoptionError(f"Recorded stage output is missing or not a regular file: {path}")
    return {
        "path": _relative_output_path(path, output_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    normalized.pop("_config_path", None)
    return normalized


def _validate_training_provenance(
    stage: str,
    repository: Path,
    output_root: Path,
) -> list[Path]:
    seed = TRAINING_SEEDS[stage]
    seed_dir = output_root / f"checkpoints/resnet50/seed_{seed:04d}"
    run_info_path = seed_dir / "run_info.json"
    config_path = seed_dir / "config_resolved.yaml"
    run_info = _read_json(run_info_path, f"training run_info for seed {seed}")
    if run_info.get("status") != "completed":
        raise AdoptionError(f"Training run_info is not completed for seed {seed}")
    if run_info.get("git_commit") != SOURCE_COMMIT:
        raise AdoptionError(f"Training run_info Git provenance mismatch for seed {seed}")
    if run_info.get("experiment") != "train_erm_resnet50":
        raise AdoptionError(f"Training run_info experiment mismatch for seed {seed}")
    if run_info.get("seed") != seed:
        raise AdoptionError(f"Training run_info seed mismatch for seed {seed}")
    recorded_hashes = [
        run_info[key]
        for key in ("protocol_hash", "protocol_sha256")
        if run_info.get(key) is not None
    ]
    if not recorded_hashes or any(value != PROTOCOL_HASH for value in recorded_hashes):
        raise AdoptionError(f"Training run_info protocol provenance mismatch for seed {seed}")
    current_config = load_config(repository / "configs/models/resnet50_hf.yaml")
    if run_info.get("dataset_version") != current_config.get("data", {}).get("version"):
        raise AdoptionError(f"Training run_info dataset version mismatch for seed {seed}")
    try:
        recorded_config = load_config(config_path)
    except (OSError, ValueError) as error:
        raise AdoptionError(f"Invalid recorded training config for seed {seed}: {error}") from error
    if _scientific_config(recorded_config) != _scientific_config(current_config):
        raise AdoptionError(f"Recorded scientific training config mismatch for seed {seed}")
    return [run_info_path, config_path]


def _original_marker(
    stage: str,
    output_root: Path,
    target_commit: str,
) -> tuple[dict[str, Any], bytes]:
    marker_path = output_root / ".stage_state" / f"{stage}.json"
    archive_path = output_root / ARCHIVE_RELATIVE_PATH / f"{stage}.json"
    marker = _read_json(marker_path, f"stage marker for {stage}")
    if marker.get("git_commit") == SOURCE_COMMIT:
        return marker, marker_path.read_bytes()
    if (
        marker.get("git_commit") == target_commit
        and marker.get("artifact_origin_git_commit") == SOURCE_COMMIT
        and marker.get("adopted_by_hotfix") is True
        and marker.get("adoption_ledger") == LEDGER_RELATIVE_PATH.as_posix()
    ):
        archived = _read_json(archive_path, f"archived original marker for {stage}")
        return archived, archive_path.read_bytes()
    raise AdoptionError(f"Stage {stage} is not an approved source marker or valid partial adoption")


def _validate_stage(
    stage: str,
    repository: Path,
    output_root: Path,
    target_commit: str,
) -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    marker, marker_bytes = _original_marker(stage, output_root, target_commit)
    if marker.get("stage") != stage:
        raise AdoptionError(f"Source stage marker names a different stage: {stage}")
    if marker.get("status") != "completed":
        raise AdoptionError(f"Source stage is not completed: {stage}")
    if marker.get("git_commit") != SOURCE_COMMIT:
        raise AdoptionError(f"Source stage Git provenance mismatch: {stage}")
    if marker.get("protocol_hash") != PROTOCOL_HASH:
        raise AdoptionError(f"Source stage protocol provenance mismatch: {stage}")
    if not marker.get("stage_signature") and not marker.get("command"):
        raise AdoptionError(f"Source stage lacks a command or stage signature: {stage}")
    outputs = marker.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise AdoptionError(f"Source stage has no recorded outputs: {stage}")
    output_paths = [Path(value) for value in outputs if isinstance(value, str)]
    if len(output_paths) != len(outputs):
        raise AdoptionError(f"Source stage has invalid output paths: {stage}")
    relative_outputs = {_relative_output_path(path, output_root) for path in output_paths}
    if relative_outputs != _expected_outputs(stage):
        raise AdoptionError(
            f"Source stage output set differs from the approved runner layout: {stage}"
        )
    provenance_paths = (
        _validate_training_provenance(stage, repository, output_root)
        if stage in TRAINING_SEEDS
        else []
    )
    artifacts = [_artifact_record(path, output_root) for path in [*output_paths, *provenance_paths]]
    return marker, marker_bytes, artifacts


def _adopted_marker(original: dict[str, Any], target_commit: str) -> dict[str, Any]:
    adopted = deepcopy(original)
    adopted.update(
        {
            "git_commit": target_commit,
            "artifact_origin_git_commit": SOURCE_COMMIT,
            "adopted_by_hotfix": True,
            "adoption_ledger": LEDGER_RELATIVE_PATH.as_posix(),
            "original_completed_at": original.get("completed_at"),
        }
    )
    return adopted


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _write_once_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise AdoptionError(f"Existing adoption provenance conflicts with validated content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _verify_existing_ledger(
    ledger: dict[str, Any],
    output_root: Path,
    target_commit: str,
    diff_paths: list[str],
) -> None:
    if ledger.get("schema_version") != 1 or ledger.get("status") != "applied":
        raise AdoptionError("Existing adoption ledger schema/status is invalid")
    expected = {
        "source_git_commit": SOURCE_COMMIT,
        "target_git_commit": target_commit,
        "protocol_hash": PROTOCOL_HASH,
        "git_diff_paths": diff_paths,
        "adopted_stages": list(ADOPTED_STAGES),
        "reason": REASON,
    }
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise AdoptionError(f"Existing adoption ledger has incompatible {key}")
    stages = ledger.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(ADOPTED_STAGES):
        raise AdoptionError("Existing adoption ledger has an invalid stage set")
    for stage, metadata in stages.items():
        if not isinstance(metadata, dict) or not isinstance(metadata.get("original_marker"), dict):
            raise AdoptionError(f"Existing adoption ledger has invalid metadata for {stage}")
        expected_archive = (ARCHIVE_RELATIVE_PATH / f"{stage}.json").as_posix()
        if metadata.get("original_marker_archive") != expected_archive:
            raise AdoptionError(f"Existing adoption ledger has an invalid archive path for {stage}")
        archive_path = output_root / expected_archive
        archived = _read_json(archive_path, f"archived original marker for {stage}")
        if archived != metadata["original_marker"]:
            raise AdoptionError(f"Archived original marker no longer matches the ledger: {stage}")
        if sha256_file(archive_path) != metadata.get("original_marker_sha256"):
            raise AdoptionError(f"Archived original marker checksum mismatch: {stage}")
        marker = _read_json(output_root / ".stage_state" / f"{stage}.json", f"adopted marker {stage}")
        if marker != _adopted_marker(metadata["original_marker"], target_commit):
            raise AdoptionError(f"Adopted stage marker no longer matches the ledger: {stage}")
        artifacts = metadata.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise AdoptionError(f"Existing adoption ledger has no artifacts for {stage}")
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                raise AdoptionError(f"Existing adoption ledger has an invalid artifact for {stage}")
            path = output_root / artifact["path"]
            if _relative_output_path(path, output_root) != artifact["path"]:
                raise AdoptionError(f"Existing adoption ledger has an unsafe artifact path for {stage}")
            if not path.is_file() or path.stat().st_size != artifact["bytes"]:
                raise AdoptionError(f"Adopted artifact size/missing-file mismatch: {path}")
            if sha256_file(path) != artifact["sha256"]:
                raise AdoptionError(f"Adopted artifact checksum mismatch: {path}")


def run_adoption(repository: Path, output_root: Path, *, apply: bool = False) -> dict[str, Any]:
    repository = repository.resolve()
    output_root = output_root.resolve()
    target_commit, diff_paths = inspect_repository(repository)
    verify_frozen_protocol(repository, output_root)
    ledger_path = output_root / LEDGER_RELATIVE_PATH
    if ledger_path.is_file():
        ledger = _read_json(ledger_path, "existing adoption ledger")
        _verify_existing_ledger(ledger, output_root, target_commit, diff_paths)
        return ledger

    stages: dict[str, Any] = {}
    original_bytes: dict[str, bytes] = {}
    originals: dict[str, dict[str, Any]] = {}
    for stage in ADOPTED_STAGES:
        original, raw_marker, artifacts = _validate_stage(
            stage, repository, output_root, target_commit
        )
        originals[stage] = original
        original_bytes[stage] = raw_marker
        stages[stage] = {
            "original_marker": original,
            "original_marker_archive": (ARCHIVE_RELATIVE_PATH / f"{stage}.json").as_posix(),
            "original_marker_sha256": sha256_file(
                output_root / ".stage_state" / f"{stage}.json"
            )
            if original.get("git_commit") == SOURCE_COMMIT
            else sha256_file(output_root / ARCHIVE_RELATIVE_PATH / f"{stage}.json"),
            "artifacts": artifacts,
        }
    ledger = {
        "schema_version": 1,
        "status": "applied" if apply else "validated",
        "reason": REASON,
        "source_git_commit": SOURCE_COMMIT,
        "target_git_commit": target_commit,
        "protocol_hash": PROTOCOL_HASH,
        "git_diff_paths": diff_paths,
        "adopted_stages": list(ADOPTED_STAGES),
        "stages": stages,
        "adoption_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not apply:
        return ledger
    ledger["status"] = "applied"
    for stage in ADOPTED_STAGES:
        archive_path = output_root / ARCHIVE_RELATIVE_PATH / f"{stage}.json"
        _write_once_or_verify(archive_path, original_bytes[stage])
    for stage in ADOPTED_STAGES:
        marker_path = output_root / ".stage_state" / f"{stage}.json"
        _write_atomic(marker_path, _json_bytes(_adopted_marker(originals[stage], target_commit)))
    _write_atomic(ledger_path, _json_bytes(ledger))
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or apply the one-time Hospital-1 audit-hotfix upstream adoption"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate only (default)")
    mode.add_argument("--apply", action="store_true", help="Explicitly write adoption provenance")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output_root = args.output_root or (
        Path(os.environ["MEDSTYLE_OUTPUT_ROOT"]) if os.environ.get("MEDSTYLE_OUTPUT_ROOT") else None
    )
    if output_root is None:
        parser.error("MEDSTYLE_OUTPUT_ROOT or --output-root is required")
    repository = Path(__file__).resolve().parents[1]
    try:
        ledger = run_adoption(repository, output_root, apply=args.apply)
    except AdoptionError as error:
        parser.error(str(error))
    mode_name = "APPLY" if args.apply else "CHECK"
    print(
        f"Hospital-1 hotfix upstream adoption {mode_name}: PASS "
        f"({len(ledger['adopted_stages'])} stages)"
    )


if __name__ == "__main__":
    main()
