"""Signed Hospital Context Effect aggregation."""

from __future__ import annotations

import pandas as pd


def directed_hce(source_table: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ["source_split", "seed", "backbone", "source_hospital", "target_hospital"] if column in source_table]
    return source_table.groupby(keys, dropna=False).agg(
        n_sources=("source_id", "nunique"),
        within_signed_shift=("within_signed_shift", "mean"),
        cross_signed_shift=("cross_signed_shift", "mean"),
        hce=("HCE_source", "mean"),
    ).reset_index()


def pair_weighted_hce(directed: pd.DataFrame, expected_pairs: set[tuple[int, int]] | None = None) -> pd.DataFrame:
    keys = [column for column in ["source_split", "seed", "backbone"] if column in directed]
    rows = []
    for values, group in directed.groupby(keys, dropna=False) if keys else [((), directed)]:
        observed = set(zip(group["source_hospital"].astype(int), group["target_hospital"].astype(int)))
        required = expected_pairs or set()
        missing = sorted(required - observed)
        unexpected = sorted(observed - required) if expected_pairs is not None else []
        available = expected_pairs is None or (not missing and not unexpected)
        record = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        record.update({
            "status": "available" if available else "unavailable",
            "hce_pair_weighted": group["hce"].mean() if available else float("nan"),
            "n_pairs": len(observed),
            "expected_pairs": sorted(required),
            "missing_pairs": missing,
            "unexpected_pairs": unexpected,
        })
        rows.append(record)
    return pd.DataFrame(rows)


def common_support_hce(source_table: pd.DataFrame, expected_targets: dict[int, set[int]] | None = None) -> pd.DataFrame:
    frame = source_table[source_table["common_support"].astype(bool)] if "common_support" in source_table else source_table
    keys = [column for column in ["source_split", "seed", "backbone"] if column in frame]
    rows = []
    for values, group in frame.groupby(keys, dropna=False) if keys else [((), frame)]:
        valid = not group.empty
        if expected_targets:
            for hospital, targets in expected_targets.items():
                sub = group[group["source_hospital"] == hospital]
                if sub.empty or set(sub["target_hospital"].astype(int).unique()) != targets:
                    valid = False
        pair_means = group.groupby(["source_hospital", "target_hospital"])["HCE_source"].mean()
        record = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        record.update(status="available" if valid else "unavailable", n_unique_sources=group["source_id"].nunique(), hce_common_support=pair_means.mean() if valid else float("nan"))
        rows.append(record)
    return pd.DataFrame(rows)
