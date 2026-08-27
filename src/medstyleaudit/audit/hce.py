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


def pair_weighted_hce(directed: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ["source_split", "seed", "backbone"] if column in directed]
    return directed.groupby(keys, dropna=False).agg(n_pairs=("hce", "size"), hce_pair_weighted=("hce", "mean")).reset_index()


def common_support_hce(source_table: pd.DataFrame) -> pd.DataFrame:
    frame = source_table[source_table["common_support"].astype(bool)] if "common_support" in source_table else source_table
    keys = [column for column in ["source_split", "seed", "backbone"] if column in frame]
    rows = []
    for values, group in frame.groupby(keys, dropna=False) if keys else [((), frame)]:
        pair_means = group.groupby(["source_hospital", "target_hospital"])["HCE_source"].mean()
        record = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        record.update(n_unique_sources=group["source_id"].nunique(), hce_common_support=pair_means.mean())
        rows.append(record)
    return pd.DataFrame(rows)
