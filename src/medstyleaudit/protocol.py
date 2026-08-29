"""Canonical final-protocol hashing and hospital-2 access lock."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import save_json

FINAL_TRAINING_SEEDS = (11, 42, 101)
FINAL_HOSPITAL1_SETTINGS = ("primary", "random_paired", "roi_only", "buffer_r8", "hard_boundary")


def resolved_protocol(path: str | Path) -> dict:
    protocol = load_config(path)
    protocol.pop("_config_path", None)
    return protocol


def protocol_sha256(path_or_protocol: str | Path | Mapping) -> str:
    protocol = resolved_protocol(path_or_protocol) if isinstance(path_or_protocol, (str, Path)) else dict(path_or_protocol)
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_protocol_lock(protocol_path: str | Path, output_path: str | Path) -> str:
    digest = protocol_sha256(protocol_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(digest + "\n", encoding="utf-8")
    return digest


def verify_protocol_lock(protocol_path: str | Path, hash_path: str | Path) -> str:
    expected = Path(hash_path).read_text(encoding="utf-8").strip()
    actual = protocol_sha256(protocol_path)
    if expected != actual:
        raise PermissionError(f"Final protocol hash mismatch: locked={expected}, current={actual}")
    return actual


def git_state(repository: str | Path = ".") -> tuple[str, bool]:
    root = str(Path(repository).resolve())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.strip())
    return commit, dirty


def _verify_current_stage_marker(path: Path, commit: str, protocol_hash: str) -> dict:
    if not path.is_file():
        raise PermissionError(f"Required Hospital-1 stage marker is missing: {path.name}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise PermissionError(f"Required Hospital-1 stage is incomplete: {path.stem}")
    if metadata.get("git_commit") != commit or metadata.get("protocol_hash") != protocol_hash:
        raise PermissionError(f"Required Hospital-1 stage has stale Git/protocol provenance: {path.stem}")
    if not metadata.get("stage_signature") and not metadata.get("command"):
        raise PermissionError(f"Required Hospital-1 stage lacks a provenance signature: {path.stem}")
    outputs = metadata.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise PermissionError(f"Required Hospital-1 stage has no recorded outputs: {path.stem}")
    missing_outputs = [output for output in outputs if not Path(output).exists()]
    if missing_outputs:
        raise PermissionError(f"Required Hospital-1 stage outputs are missing for {path.stem}: {missing_outputs}")
    return metadata


def verify_final_test_prerequisites(
    output_root: str | Path,
    protocol_path: str | Path = "configs/final/FINAL_PROTOCOL.yaml",
    repository: str | Path = ".",
) -> dict:
    """Verify persisted Hospital-1 completion for the current commit and protocol."""
    root = Path(output_root)
    protocol_hash = verify_protocol_lock(protocol_path, root / "protocol" / "protocol_sha256.txt")
    commit, dirty = git_state(repository)
    if dirty:
        raise PermissionError("Final test prerequisites require a clean Git working tree")
    stage_root = root / ".stage_state"

    training = [f"train_resnet50_{seed:04d}" for seed in FINAL_TRAINING_SEEDS]
    audits = []
    for seed in FINAL_TRAINING_SEEDS:
        audits.extend(f"audit_hospital1_{setting}_{seed:04d}" for setting in ("primary", "random_paired", "roi_only"))
        audits.extend(f"robustness_hospital1_{setting}_{seed:04d}" for setting in ("buffer_r8", "hard_boundary"))
    for stage in [*training, *audits, "aggregate_hospital1"]:
        _verify_current_stage_marker(stage_root / f"{stage}.json", commit, protocol_hash)

    completeness_path = root / "aggregate" / "hospital1" / "run_completeness.csv"
    if not completeness_path.is_file():
        raise PermissionError("Hospital-1 aggregation is incomplete: run_completeness.csv is missing")
    with completeness_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_setting = {row.get("setting"): row for row in rows if row.get("population") == "hospital1"}
    missing_settings = set(FINAL_HOSPITAL1_SETTINGS) - set(by_setting)
    if missing_settings:
        raise PermissionError(f"Hospital-1 aggregation is missing required settings: {sorted(missing_settings)}")
    incomplete = []
    for setting in FINAL_HOSPITAL1_SETTINGS:
        row = by_setting[setting]
        try:
            expected = int(row["expected_runs"])
            completed = int(row["completed_runs"])
        except (KeyError, TypeError, ValueError) as error:
            raise PermissionError(f"Hospital-1 completeness row is invalid for {setting}") from error
        if expected != len(FINAL_TRAINING_SEEDS) or completed != expected or row.get("status") != "complete":
            incomplete.append(setting)
    if incomplete:
        raise PermissionError(f"Hospital-1 aggregation is incomplete for settings: {incomplete}")
    return {"protocol_hash": protocol_hash, "git_commit": commit}


def authorize_final_test(output_root: str | Path, protocol_path: str | Path = "configs/final/FINAL_PROTOCOL.yaml") -> dict:
    root = Path(output_root)
    preflight_path = root / "protocol" / "final_preflight.json"
    hash_path = root / "protocol" / "protocol_sha256.txt"
    unlock_path = root / "protocol" / "final_test_lock.json"
    if not Path(protocol_path).is_file() or not hash_path.is_file() or not preflight_path.is_file() or not unlock_path.is_file():
        raise PermissionError("Final test is locked: protocol, protocol hash, PASS preflight, and explicit unlock are required")
    protocol_hash = verify_protocol_lock(protocol_path, hash_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("protocol_hash") != protocol_hash:
        raise PermissionError("Final test is locked: final preflight is not PASS for the current protocol")
    commit, dirty = git_state()
    if dirty or preflight.get("git_commit") != commit:
        raise PermissionError("Final test is locked: Git must be clean and match the preflight commit")
    unlock = json.loads(unlock_path.read_text(encoding="utf-8"))
    if (
        unlock.get("status") != "unlocked"
        or unlock.get("protocol_sha256") != protocol_hash
        or unlock.get("git_commit") != commit
    ):
        raise PermissionError("Final test is locked: explicit unlock does not match the current protocol and commit")
    return {"protocol_hash": protocol_hash, "git_commit": commit}


def record_final_test_opened(output_root: str | Path, run_id: str, operator: str | None = None) -> Path:
    root = Path(output_root)
    authorization = authorize_final_test(root)
    path = root / "protocol" / "final_test_opened.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **authorization,
        "operator": operator or os.environ.get("MEDSTYLE_OPERATOR") or os.environ.get("USER") or "unknown",
        "run_id": run_id,
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol_hash") != payload["protocol_hash"] or existing.get("git_commit") != payload["git_commit"]:
            raise PermissionError("Existing final-test marker belongs to another protocol or commit")
        return path
    return save_json(payload, path)


def assert_frozen_execution_config(
    *,
    protocol_path: str | Path = "configs/final/FINAL_PROTOCOL.yaml",
    matching: Mapping | None = None,
    model: Mapping | None = None,
    audit: Mapping | None = None,
    robustness: Mapping | None = None,
    seed: int | None = None,
) -> None:
    """Reject final-test configs that diverge from locked protocol fields."""
    protocol = resolved_protocol(protocol_path)
    if matching is not None:
        keys = [
            "donors_per_source", "lambda_balance", "lambda_pair", "tau_distance",
            "tau_balance", "candidate_pool_size", "donor_reuse_cap",
            "donor_slide_reuse_cap", "min_donor_slide_diversity", "exact_columns",
        ]
        mismatches = [key for key in keys if matching.get(key) != protocol["matching"].get(key)]
        if matching.get("method") != protocol["matching"]["method"]: mismatches.append("method")
        if list(matching.get("descriptor_columns", [])) != list(protocol["matching"]["descriptor_features"]): mismatches.append("descriptor_columns")
        if matching.get("physical_id_column") != "physical_id": mismatches.append("physical_id_column")
        if dict(matching.get("feature_calipers", {})) != {}: mismatches.append("feature_calipers")
        configured_test_targets = {int(value) for value in matching.get("target_hospitals", {}).get("test", [])}
        frozen_test_targets = {int(pair[1]) for pair in protocol["audit"]["expected_directed_pairs"]["test"]}
        if configured_test_targets != frozen_test_targets: mismatches.append("target_hospitals.test")
        if mismatches: raise PermissionError(f"Final-test matching config diverges from the frozen protocol: {mismatches}")
    if model is not None:
        architecture = model.get("model", {}).get("architecture")
        training = model.get("training", {})
        mismatches = []
        if architecture not in protocol["models"]["backbones"]: mismatches.append("architecture")
        if bool(model.get("model", {}).get("pretrained", False)): mismatches.append("pretrained")
        checks = {"epochs": "epochs", "batch_size": "batch_size", "optimizer": "optimizer", "scheduler": "scheduler", "learning_rate": "learning_rate", "weight_decay": "weight_decay"}
        for source, target in checks.items():
            if str(training.get(source)).lower() != str(protocol["models"][target]).lower(): mismatches.append(source)
        if seed is not None and int(seed) not in protocol["models"]["seeds"]: mismatches.append("seed")
        if training.get("selection_metric") != protocol["model"]["selection_metric"]: mismatches.append("selection_metric")
        if not bool(training.get("maximize_metric", False)): mismatches.append("maximize_metric")
        if mismatches: raise PermissionError(f"Final-test model config diverges from the frozen protocol: {mismatches}")
    if audit is not None:
        expected = protocol["scientific_definitions"]
        mismatches = [name for name in ("roi_size", "patch_size") if int(audit.get(name, -1)) != int(expected[name])]
        audit_checks = {
            "source_buffer": protocol["audit"]["primary_source_buffer"],
            "feather_width": protocol["audit"]["feather_width"],
            "inference_batch_size": protocol["audit"]["inference_batch_size"],
            "q_min": protocol["statistics"]["q_min"],
            "bootstrap_draws": protocol["statistics"]["bootstrap_draws"],
            "confidence": protocol["statistics"]["confidence_level"],
            "audit_sample_size": protocol["audit"]["audit_sample_size"],
            "audit_sampling_seed": protocol["audit"]["audit_sampling_seed"],
            "pair_weights": "equal",
        }
        mismatches.extend(name for name, value in audit_checks.items() if audit.get(name) != value)
        configured = {tuple(map(int, pair)) for pair in audit.get("expected_directed_pairs", {}).get("test", [])}
        frozen = {tuple(map(int, pair)) for pair in protocol["audit"]["expected_directed_pairs"]["test"]}
        if configured != frozen: mismatches.append("expected_directed_pairs.test")
        if mismatches: raise PermissionError(f"Final-test audit config diverges from the frozen protocol: {mismatches}")
    if robustness is not None:
        actual = (
            int(robustness.get("source_buffer", -1)),
            str(robustness.get("boundary_mode", "")),
            int(robustness.get("feather_width", -1)),
        )
        width = int(protocol["audit"]["feather_width"])
        allowed = {
            (int(protocol["audit"]["primary_source_buffer"]), str(protocol["audit"]["primary_boundary"]), width),
            (int(protocol["audit"]["robustness_source_buffer"]), str(protocol["audit"]["primary_boundary"]), width),
            (int(protocol["audit"]["primary_source_buffer"]), str(protocol["audit"]["robustness_boundary"]), 0),
        }
        if actual not in allowed:
            raise PermissionError(f"Final-test robustness setting diverges from the frozen protocol: {actual}")
