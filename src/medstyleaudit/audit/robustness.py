"""Aggregate robustness grids while preserving setting-specific coverage."""

from __future__ import annotations

import pandas as pd


def robustness_summary(source_metrics: pd.DataFrame, setting_columns: list[str]) -> pd.DataFrame:
    keys = setting_columns + [column for column in ["source_split", "seed", "backbone", "source_hospital", "target_hospital"] if column in source_metrics]
    return source_metrics.groupby(keys, dropna=False).agg(n_sources=("source_id", "nunique"), hcs=("HCS_source", "mean"), hce=("HCE_source", "mean"), delta_within=("delta_within", "mean"), delta_cross=("delta_cross", "mean")).reset_index()
