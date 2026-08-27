"""Hospital Context Sensitivity aggregation."""

from __future__ import annotations

import pandas as pd


def directed_hcs(source_table: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ["source_split", "seed", "backbone", "source_hospital", "target_hospital"] if column in source_table]
    return source_table.groupby(keys, dropna=False).agg(
        n_sources=("source_id", "nunique"),
        delta_within=("delta_within", "mean"),
        delta_cross=("delta_cross", "mean"),
        hcs=("HCS_source", "mean"),
    ).reset_index()


def pair_weighted_hcs(directed: pd.DataFrame, expected_pairs: set[tuple[int, int]] | None = None) -> pd.DataFrame:
    grouping = [column for column in ["source_split", "seed", "backbone"] if column in directed]
    rows = []
    for keys, group in directed.groupby(grouping, dropna=False) if grouping else [((), directed)]:
        observed = set(zip(group["source_hospital"].astype(int), group["target_hospital"].astype(int)))
        available = not expected_pairs or observed == expected_pairs
        record = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        record.update({"status": "available" if available else "unavailable", "hcs_pair_weighted": group["hcs"].mean() if available else float("nan"), "n_pairs": len(group), "missing_pairs": sorted((expected_pairs or set()) - observed)})
        rows.append(record)
    return pd.DataFrame(rows)


def common_support_hcs(source_table: pd.DataFrame, expected_targets: dict[int, set[int]] | None = None) -> pd.DataFrame:
    frame = source_table.copy()
    if "common_support" in frame:
        frame = frame[frame["common_support"].astype(bool)]
    grouping = [column for column in ["source_split", "seed", "backbone"] if column in frame]
    rows = []
    for keys, group in frame.groupby(grouping, dropna=False) if grouping else [((), frame)]:
        valid = not group.empty
        if expected_targets:
            for hospital, targets in expected_targets.items():
                sub = group[group["source_hospital"] == hospital]
                if sub.empty or set(sub["target_hospital"].astype(int).unique()) != targets:
                    valid = False
        record = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        pair_means = group.groupby(["source_hospital", "target_hospital"], dropna=False).agg(
            delta_within=("delta_within", "mean"), delta_cross=("delta_cross", "mean"), hcs=("HCS_source", "mean")
        )
        record.update({
            "status": "available" if valid else "unavailable",
            "n_unique_sources": group["source_id"].nunique(),
            "delta_within": pair_means["delta_within"].mean() if valid else float("nan"),
            "delta_cross": pair_means["delta_cross"].mean() if valid else float("nan"),
            "hcs_common_support": pair_means["hcs"].mean() if valid else float("nan"),
        })
        rows.append(record)
    return pd.DataFrame(rows)
