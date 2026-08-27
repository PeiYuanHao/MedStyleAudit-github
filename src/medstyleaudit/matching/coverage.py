"""Coverage, common-support, and attrition reporting."""

from __future__ import annotations

import pandas as pd


def directed_coverage(ledger: pd.DataFrame) -> pd.DataFrame:
    required = {"source_id", "source_split", "source_hospital", "target_hospital", "coverage_status"}
    if not required.issubset(ledger):
        raise ValueError(f"Coverage ledger missing columns: {sorted(required - set(ledger))}")
    grouped = ledger.groupby(["source_split", "source_hospital", "target_hospital"], dropna=False)
    rows = []
    for keys, group in grouped:
        accepted = int((group["coverage_status"] == "accepted").sum())
        rows.append(dict(zip(["source_split", "source_hospital", "target_hospital"], keys), candidate_sources=len(group), accepted_sources=accepted, coverage=accepted / len(group)))
    return pd.DataFrame(rows)


def annotate_common_support(ledger: pd.DataFrame) -> pd.DataFrame:
    result = ledger.copy()
    key = ["source_split", "source_hospital", "source_id"]
    common = result.groupby(key)["coverage_status"].transform(lambda values: bool((values == "accepted").all()))
    result["common_support"] = common
    return result


def common_support_coverage(ledger: pd.DataFrame) -> pd.DataFrame:
    annotated = annotate_common_support(ledger)
    sources = annotated.drop_duplicates(["source_split", "source_hospital", "source_id"])
    return sources.groupby(["source_split", "source_hospital"])["common_support"].agg(candidate_sources="size", accepted_sources="sum", coverage="mean").reset_index()


def attrition_table(ledger: pd.DataFrame) -> pd.DataFrame:
    result = ledger.copy()
    result["exclusion_reason"] = result["exclusion_reason"].fillna("accepted")
    return result.groupby(["source_split", "source_hospital", "target_hospital", "exclusion_reason"]).size().rename("count").reset_index()
