"""Append unlocked hospital-2 matching to P0 and regenerate all diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from medstyleaudit.matching.balance import (
    balance_summary,
    donor_reuse_detail,
    donor_reuse_distribution,
    feature_balance,
    slide_reuse_detail,
    slide_reuse_distribution,
)
from medstyleaudit.matching.coverage import (
    attrition_table,
    common_support_coverage,
    directed_coverage,
)
from medstyleaudit.matching.distance import Standardizer
from medstyleaudit.utils.cli import enforce_final_test_guard
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_json, save_table
from medstyleaudit.utils.paths import experiment_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, default=None)
    parser.add_argument("--final-test", type=Path, default=None)
    parser.add_argument("--config", default="configs/matching/primary.yaml")
    parser.add_argument("--allow-final-test", action="store_true")
    args = parser.parse_args()
    enforce_final_test_guard("test", args.allow_final_test)
    primary = args.primary or experiment_path("p0/matching")
    final_test = args.final_test or experiment_path("p0/matching_final_test")
    config = load_config(args.config); features = list(config["matching"]["descriptor_columns"])
    descriptors = read_table(config["data"]["descriptors"])
    standardizer_path = primary / "descriptor_standardizer.json"
    if not standardizer_path.is_file():
        raise FileNotFoundError(f"Training-fitted descriptor standardizer is missing: {standardizer_path}")
    standardizer = Standardizer.from_dict(json.loads(standardizer_path.read_text(encoding="utf-8")))
    triplets = pd.concat([read_table(primary / "triplets.parquet").query("source_split != 'test'"), read_table(final_test / "triplets.parquet")], ignore_index=True)
    ledger = pd.concat([read_table(primary / "source_ledger.parquet").query("source_split != 'test'"), read_table(final_test / "source_ledger.parquet")], ignore_index=True)
    save_table(triplets, primary / "triplets.parquet"); save_table(ledger, primary / "source_ledger.parquet")
    save_table(directed_coverage(ledger), primary / "directed_coverage.csv")
    save_table(common_support_coverage(ledger), primary / "common_support_coverage.csv")
    save_table(attrition_table(ledger), primary / "attrition.csv")
    save_table(balance_summary(triplets), primary / "matching_balance.csv")
    save_table(feature_balance(triplets, descriptors, features, standardizer), primary / "feature_balance.csv")
    save_table(donor_reuse_detail(triplets, descriptors), primary / "donor_reuse.csv")
    save_table(donor_reuse_distribution(triplets, descriptors), primary / "donor_reuse_summary.csv")
    save_table(slide_reuse_detail(triplets, descriptors), primary / "slide_reuse.csv")
    save_table(slide_reuse_distribution(triplets, descriptors), primary / "slide_reuse_summary.csv")
    directed = directed_coverage(ledger); common = common_support_coverage(ledger)
    test_directed = directed[directed["source_split"] == "test"]
    test_common = common[common["source_split"] == "test"]
    feature = feature_balance(triplets, descriptors, features, standardizer)
    test_feature = feature[feature["source_split"] == "test"]
    reuse = donor_reuse_detail(triplets, descriptors)
    donor_rows = reuse
    checks = {
        "directed_coverage": float(test_directed["coverage"].min()) >= .90 if not test_directed.empty else False,
        "common_support_coverage": float(test_common["coverage"].min()) >= .90 if not test_common.empty else False,
        "per_feature_abs_smd": bool(not test_feature.empty and test_feature["paired_smd"].notna().all() and test_feature["paired_smd"].abs().max() <= .10),
        "donor_reuse_cap": int(donor_rows["total_uses"].max()) <= int(config["matching"]["donor_reuse_cap"]),
    }
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    save_json(report, primary / "final_matching_check.json")
    if report["status"] != "PASS":
        raise SystemExit(f"Final-test matching check FAIL: {[name for name, passed in checks.items() if not passed]}")


if __name__ == "__main__":
    main()
