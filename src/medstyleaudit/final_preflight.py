"""Automated, fail-closed authorization gate for the frozen final suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from medstyleaudit.protocol import git_state, verify_protocol_lock
from medstyleaudit.utils.io import read_table, save_json


def _record(checks: list[dict], name: str, passed: bool, **details: Any) -> None:
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", **details})


def run_final_preflight(specification: Mapping[str, Any], output_path: str | Path) -> dict:
    """Evaluate every frozen preflight condition and always write a JSON report."""
    checks: list[dict] = []
    files = {name: Path(path) for name, path in specification["files"].items()}
    required = specification.get("required_matching_files", ["descriptors", "directed_coverage", "common_support_coverage", "matching_balance", "feature_balance", "donor_reuse", "slide_reuse", "source_ledger"])
    for name in required:
        path = files.get(name)
        _record(checks, f"required_file:{name}", bool(path and path.is_file()), path=str(path) if path else None)

    pytest_marker = Path(specification["pytest_marker"])
    pytest_data = json.loads(pytest_marker.read_text(encoding="utf-8")) if pytest_marker.is_file() else {}
    _record(checks, "unit_tests", pytest_data.get("status") == "PASS")
    _record(checks, "roi_tests", pytest_data.get("roi_tests") == "PASS")

    dataset_files = [Path(path) for path in specification.get("required_dataset_files", [])]
    _record(checks, "dataset_files", bool(dataset_files) and all(path.exists() for path in dataset_files), missing=[str(path) for path in dataset_files if not path.exists()])

    integrity_path = files.get("integrity_report")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8")) if integrity_path and integrity_path.is_file() else {}
    reported_integrity = integrity.get("status", integrity.get("overall_status"))
    integrity_ok = not integrity.get("fatal", False) and str(reported_integrity or "").lower() not in {"fail", "failed", "fatal"}
    _record(checks, "metadata_integrity", bool(integrity) and integrity_ok, reported_status=reported_integrity)

    if files.get("descriptors") and files["descriptors"].is_file():
        descriptors = read_table(files["descriptors"])
        columns = list(specification["descriptor_features"])
        finite = not descriptors.empty and set(columns) <= set(descriptors) and np.isfinite(descriptors[columns].to_numpy(dtype=float)).all()
        _record(checks, "finite_descriptors", finite, rows=len(descriptors))

    thresholds = specification["thresholds"]
    if files.get("directed_coverage") and files["directed_coverage"].is_file():
        directed = read_table(files["directed_coverage"])
        minimum = float(directed["coverage"].min()) if not directed.empty else 0.0
        _record(checks, "directed_coverage", minimum >= float(thresholds["directed_coverage"]), observed=minimum)
        actual_pairs = set(zip(directed["source_split"].astype(str), directed["source_hospital"].astype(int), directed["target_hospital"].astype(int)))
        expected_pairs = {tuple([str(row[0]), int(row[1]), int(row[2])]) for row in specification["expected_hospital_pairs"]}
        _record(checks, "expected_hospital_pairs", actual_pairs == expected_pairs, missing=sorted(expected_pairs - actual_pairs), unexpected=sorted(actual_pairs - expected_pairs))
    if files.get("common_support_coverage") and files["common_support_coverage"].is_file():
        common = read_table(files["common_support_coverage"])
        minimum = float(common["coverage"].min()) if not common.empty else 0.0
        _record(checks, "common_support_coverage", minimum >= float(thresholds["common_support_coverage"]), observed=minimum)
    if files.get("matching_balance") and files["matching_balance"].is_file():
        balance = read_table(files["matching_balance"])
        aggregate = float(np.average(np.abs(balance["paired_smd"]), weights=balance.get("n_triplets", pd.Series(1, index=balance.index)))) if not balance.empty else float("inf")
        _record(checks, "aggregate_matching_smd", np.isfinite(aggregate) and aggregate <= float(thresholds["aggregate_abs_paired_smd"]), observed=aggregate)
    if files.get("feature_balance") and files["feature_balance"].is_file():
        feature = read_table(files["feature_balance"])
        primary = feature[feature["feature"].isin(specification["descriptor_features"])]
        finite = not primary.empty and np.isfinite(primary["paired_smd"].to_numpy(dtype=float)).all()
        maximum = float(np.abs(primary["paired_smd"]).max()) if finite else float("inf")
        _record(checks, "per_feature_balance", finite and maximum <= float(thresholds["per_feature_abs_smd"]), observed=maximum)
    if files.get("donor_reuse") and files["donor_reuse"].is_file():
        reuse = read_table(files["donor_reuse"])
        detail = reuse[reuse["row_type"] == "donor"] if "row_type" in reuse else reuse
        usage_column = "total_uses" if "total_uses" in detail else "total_reuse"
        maximum = int(detail[usage_column].max()) if not detail.empty else 0
        _record(checks, "donor_reuse_cap", maximum <= int(specification["donor_reuse_cap"]), observed=maximum)

    lesion_path = files.get("alignment_report")
    lesion = json.loads(lesion_path.read_text(encoding="utf-8")) if lesion_path and lesion_path.is_file() else {}
    lesion_status = "available" if str(lesion.get("status", "")).lower() in {"pass", "passed", "validated", "available"} else "unavailable"

    try:
        protocol_hash = verify_protocol_lock(specification["protocol_path"], specification["protocol_hash_path"])
        _record(checks, "protocol_hash", True, value=protocol_hash)
    except Exception as error:
        protocol_hash = None
        _record(checks, "protocol_hash", False, reason=str(error))
    try:
        commit, dirty = git_state(specification.get("repository", "."))
        _record(checks, "git_clean", not dirty, git_commit=commit)
    except Exception as error:
        commit, dirty = None, True
        _record(checks, "git_clean", False, reason=str(error))

    report = {
        "status": "PASS" if checks and all(item["status"] == "PASS" for item in checks) else "FAIL",
        "protocol_hash": protocol_hash, "git_commit": commit, "lesion_aware_status": lesion_status,
        "checks": checks,
    }
    save_json(report, output_path)
    return report
