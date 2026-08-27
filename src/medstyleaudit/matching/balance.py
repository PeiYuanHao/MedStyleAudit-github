"""Paired matching-balance diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def balance_summary(triplets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in triplets.groupby(["source_split", "source_hospital", "target_hospital"]):
        within = group["matching_distance_within"].to_numpy(float)
        cross = group["matching_distance_cross"].to_numpy(float)
        scale = float(np.std(np.concatenate([within, cross]), ddof=1)) if len(group) > 1 else 0.0
        rows.append({
            "source_split": keys[0], "source_hospital": keys[1], "target_hospital": keys[2],
            "n_triplets": len(group), "within_mean": within.mean(), "cross_mean": cross.mean(),
            "within_median": np.median(within), "cross_median": np.median(cross),
            "mean_paired_difference": np.mean(cross - within),
            "paired_smd": (cross.mean() - within.mean()) / scale if scale > 0 else (0.0 if np.array_equal(within, cross) else np.nan),
            "mean_pair_distance": group["matching_distance_pair"].mean(),
        })
    return pd.DataFrame(rows)
