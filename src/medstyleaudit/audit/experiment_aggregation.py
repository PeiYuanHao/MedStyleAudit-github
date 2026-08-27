"""Reproducible aggregation across prespecified seeds and backbones."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import yaml

from medstyleaudit.utils.io import save_table


RUN_TABLES = (
    "directed_hcs",
    "directed_hce",
    "global_hcs_pair_weighted",
    "global_hcs_common_support",
    "global_hce_pair_weighted",
    "global_hce_common_support",
    "directed_intervals",
)


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _compatible_run(
    directory: Path,
    *,
    architecture: str,
    seed: int,
    split: str,
    dataset_version: str,
    audit_config: Mapping,
    expected_pairs: set[tuple[int, int]],
) -> tuple[dict[str, pd.DataFrame] | None, str | None]:
    info_path, config_path = directory / "run_info.json", directory / "config_resolved.yaml"
    if not info_path.is_file() or not config_path.is_file():
        return None, "run_info.json or config_resolved.yaml is missing"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    config = _read_yaml(config_path)
    if info.get("status") != "completed":
        return None, f"run status is {info.get('status', 'missing')}"
    if int(info.get("seed", -1)) != seed:
        return None, f"run seed {info.get('seed')} does not match expected seed {seed}"
    if info.get("dataset_version") != dataset_version:
        return None, f"dataset version {info.get('dataset_version')} does not match {dataset_version}"
    if config.get("model_architecture") != architecture:
        return None, f"model architecture {config.get('model_architecture')} does not match {architecture}"
    if config.get("audit") != dict(audit_config):
        return None, "audit configuration is incompatible with the aggregation config"
    configured_pairs = config.get("audit", {}).get("expected_directed_pairs", {}).get(split, [])
    if {tuple(map(int, pair)) for pair in configured_pairs} != expected_pairs:
        return None, "configured expected hospital pairs are inconsistent"
    missing_files = [name for name in RUN_TABLES if not (directory / f"{name}.csv").is_file()]
    if missing_files:
        return None, f"required result tables are missing: {missing_files}"
    tables = {}
    for name in RUN_TABLES:
        try:
            tables[name] = pd.read_csv(directory / f"{name}.csv")
        except pd.errors.EmptyDataError:
            tables[name] = pd.DataFrame()
    for name in ("directed_hcs", "directed_hce"):
        table = tables[name]
        if table.empty:
            return None, f"{name} is empty"
        if set(table.get("source_split", pd.Series(dtype=str)).astype(str)) != {split}:
            return None, f"{name} split is inconsistent with {split}"
        if set(table.get("seed", pd.Series(dtype=int)).astype(int)) != {seed}:
            return None, f"{name} seed is inconsistent with {seed}"
        if set(table.get("backbone", pd.Series(dtype=str)).astype(str)) != {architecture}:
            return None, f"{name} backbone is inconsistent with {architecture}"
    return tables, None


def _combine(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _seed_stability(directed_hcs: pd.DataFrame, directed_hce: pd.DataFrame, expected_counts: Mapping[str, int]) -> pd.DataFrame:
    frames = []
    if not directed_hcs.empty:
        frames.append(directed_hcs.rename(columns={"hcs": "value"}).assign(metric="directed_hcs"))
    if not directed_hce.empty:
        frames.append(directed_hce.rename(columns={"hce": "value"}).assign(metric="directed_hce"))
    if not frames:
        return pd.DataFrame(columns=["metric", "source_split", "backbone", "source_hospital", "target_hospital", "n_seeds", "expected_seeds", "status", "mean", "std", "min", "max"])
    long = pd.concat(frames, ignore_index=True)
    keys = ["metric", "source_split", "backbone", "source_hospital", "target_hospital"]
    result = long.groupby(keys, dropna=False)["value"].agg(n_seeds="count", mean="mean", std="std", min="min", max="max").reset_index()
    result["expected_seeds"] = result["backbone"].map(expected_counts)
    result["status"] = result.apply(lambda row: "available" if row.n_seeds == row.expected_seeds else "unavailable", axis=1)
    return result


def aggregate_experiment_runs(
    audit_root: str | Path,
    output_dir: str | Path,
    expected_runs: Mapping[str, Mapping[str, object]],
    *,
    split: str,
    audit_config: Mapping,
    expected_pairs: set[tuple[int, int]],
    coverage_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Aggregate only compatible completed runs and ledger every absent/invalid run."""
    root, output = Path(audit_root), Path(output_dir)
    collected: dict[str, list[pd.DataFrame]] = {name: [] for name in RUN_TABLES}
    missing_rows = []
    for architecture, specification in expected_runs.items():
        dataset_version = str(specification["dataset_version"])
        for seed in map(int, specification["seeds"]):
            directory = root / architecture / f"seed_{seed:04d}" / split
            if not directory.is_dir():
                missing_rows.append({"backbone": architecture, "seed": seed, "split": split, "status": "missing", "reason": "expected run directory is missing", "run_path": str(directory)})
                continue
            tables, reason = _compatible_run(directory, architecture=architecture, seed=seed, split=split, dataset_version=dataset_version, audit_config=audit_config, expected_pairs=expected_pairs)
            if reason:
                missing_rows.append({"backbone": architecture, "seed": seed, "split": split, "status": "invalid", "reason": reason, "run_path": str(directory)})
                continue
            for name, table in tables.items():
                collected[name].append(table)

    directed_hcs = _combine(collected["directed_hcs"])
    directed_hce = _combine(collected["directed_hce"])
    global_hcs = _combine([
        *[frame.assign(summary_type="pair_weighted") for frame in collected["global_hcs_pair_weighted"]],
        *[frame.assign(summary_type="common_support") for frame in collected["global_hcs_common_support"]],
    ])
    global_hce = _combine([
        *[frame.assign(summary_type="pair_weighted") for frame in collected["global_hce_pair_weighted"]],
        *[frame.assign(summary_type="common_support") for frame in collected["global_hce_common_support"]],
    ])
    coverage_frames = []
    coverage_path = Path(coverage_dir) if coverage_dir else None
    if coverage_path:
        for name, summary_type in (("directed_coverage.csv", "directed"), ("common_support_coverage.csv", "common_support")):
            path = coverage_path / name
            if path.is_file():
                frame = pd.read_csv(path)
                if "source_split" in frame:
                    frame = frame[frame["source_split"].astype(str) == split]
                coverage_frames.append(frame.assign(summary_type=summary_type, status="available"))
    if not coverage_frames:
        coverage_frames = [pd.DataFrame([{"source_split": split, "summary_type": "all", "status": "unavailable", "reason": "coverage tables are missing"}])]
    expected_counts = {architecture: len(specification["seeds"]) for architecture, specification in expected_runs.items()}
    outputs = {
        "combined_directed_hcs": directed_hcs,
        "combined_directed_hce": directed_hce,
        "combined_global_hcs": global_hcs,
        "combined_global_hce": global_hce,
        "combined_coverage": _combine(coverage_frames),
        "combined_directed_intervals": _combine(collected["directed_intervals"]),
        "seed_stability": _seed_stability(directed_hcs, directed_hce, expected_counts),
        "missing_runs": pd.DataFrame(missing_rows, columns=["backbone", "seed", "split", "status", "reason", "run_path"]),
    }
    for name, frame in outputs.items():
        save_table(frame, output / f"{name}.csv")
    return outputs
