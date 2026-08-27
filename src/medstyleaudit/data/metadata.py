"""Canonical metadata tables and integrity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

CANONICAL_COLUMNS = ["source_id", "split", "hospital_id", "slide_id", "patient_id", "label"]


def canonicalize_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common WILDS metadata column names without guessing identities."""
    aliases = {
        "source_id": ["source_id", "patch_id", "id", "index"],
        "split": ["split", "split_name"],
        "hospital_id": ["hospital_id", "hospital", "center"],
        "slide_id": ["slide_id", "slide", "wsi"],
        "patient_id": ["patient_id", "patient"],
        "label": ["label", "y", "target"],
    }
    selected: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        found = [column for column in candidates if column in frame.columns]
        if found:
            selected[found[0]] = canonical
    result = frame.rename(columns=selected).copy()
    for required in ["source_id", "split", "hospital_id", "slide_id", "label"]:
        if required not in result:
            raise ValueError(f"Metadata column is required and could not be resolved: {required}")
    if "patient_id" not in result:
        result["patient_id"] = pd.NA
    result["physical_id"] = result["patient_id"].astype("string")
    missing_patient = result["patient_id"].isna() | result["patient_id"].astype(str).isin(["", "nan", "<NA>"])
    result.loc[missing_patient, "physical_id"] = "slide:" + result.loc[missing_patient, "slide_id"].astype(str)
    result.loc[~missing_patient, "physical_id"] = "patient:" + result.loc[~missing_patient, "patient_id"].astype(str)
    if result["source_id"].duplicated().any():
        raise ValueError("source_id must be unique")
    return result


def metadata_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return machine-readable patch counts by split/hospital/slide/label."""
    return (
        frame.groupby(["split", "hospital_id", "slide_id", "label"], dropna=False)
        .size()
        .rename("patch_count")
        .reset_index()
        .sort_values(["split", "hospital_id", "slide_id", "label"])
    )


def validate_integrity(frame: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate split/hospital consistency and physical cluster identity."""
    expected = config.get("data", config)
    split_hospitals = expected.get("split_hospitals", {})
    consistency: list[dict[str, Any]] = []
    for split, hospitals in split_hospitals.items():
        observed = sorted(frame.loc[frame["split"] == split, "hospital_id"].dropna().astype(int).unique().tolist())
        consistency.append({"split": split, "expected": sorted(hospitals), "observed": observed, "pass": observed == sorted(hospitals)})
    physical_conflicts = (
        frame.groupby("physical_id")["hospital_id"].nunique(dropna=True).loc[lambda x: x > 1].index.tolist()
    )
    checks = {
        "expected_patch_count": {
            "status": "PASS" if len(frame) == int(expected.get("expected_patches", len(frame))) else "FAIL",
            "expected": int(expected.get("expected_patches", len(frame))),
            "observed": len(frame),
        },
        "expected_wsi_count": {
            "status": "PASS" if frame["slide_id"].nunique() == int(expected.get("expected_slides", frame["slide_id"].nunique())) else "FAIL",
            "expected": int(expected.get("expected_slides", frame["slide_id"].nunique())),
            "observed": int(frame["slide_id"].nunique()),
        },
        "split_hospital_consistency": {
            "status": "PASS" if all(row["pass"] for row in consistency) else "FAIL",
            "details": consistency,
        },
        "physical_cluster_identity": {
            "status": "PASS" if not physical_conflicts else "FAIL",
            "conflicting_ids": physical_conflicts,
            "patient_available_fraction": float(frame["patient_id"].notna().mean()),
            "fallback": "slide when patient unavailable",
        },
    }
    checks["overall_status"] = "PASS" if all(v.get("status") == "PASS" for v in checks.values() if isinstance(v, dict)) else "FAIL"
    return checks


def load_metadata_csv(path: str | Path) -> pd.DataFrame:
    return canonicalize_metadata(pd.read_csv(path))
