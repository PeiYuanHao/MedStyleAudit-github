"""Fixed joint-cell quota assignment for context-randomized training."""

from __future__ import annotations

import numpy as np
import pandas as pd


def largest_remainder_counts(probabilities: pd.Series, total: int) -> pd.Series:
    raw = probabilities / probabilities.sum() * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    order = (raw - counts).sort_values(ascending=False, kind="mergesort").index[:remainder]
    counts.loc[order] += 1
    return counts


def joint_cell_quotas(donor_bank: pd.DataFrame, n: int) -> pd.DataFrame:
    cells = donor_bank.groupby(["hospital_id", "label"]).size().rename("available")
    counts = largest_remainder_counts(cells.astype(float), n)
    return pd.DataFrame({"quota": counts, "available": cells}).reset_index()
