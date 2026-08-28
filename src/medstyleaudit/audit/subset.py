"""Deterministic compute-efficient audit source subset selection.

The final suite fixes a single audit subset per held-out hospital so that the
primary Balanced audit, Random Paired, ROI-only, r=8 buffer, hard boundary, and
primary feathered boundary all reuse the *same* selected source IDs. Selection
runs after matching eligibility and before any classifier outcome is read, and is
a pure function of the eligible sources plus the sampling seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def matched_eligible_sources(triplets: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return one row per matched eligible source for a given split."""
    if triplets.empty:
        return triplets.iloc[:0].copy()
    required = {"source_id", "source_split", "source_hospital", "label", "source_slide"}
    missing = required - set(triplets.columns)
    if missing:
        raise KeyError(f"Triplet table is missing columns required for subset selection: {sorted(missing)}")
    columns = [column for column in ["source_id", "source_split", "source_hospital", "source_slide", "source_physical_id", "label"] if column in triplets]
    eligible = triplets[triplets["source_split"] == split][columns].drop_duplicates("source_id")
    return eligible.sort_values("source_id").reset_index(drop=True)


def select_audit_subset(
    sources: pd.DataFrame,
    *,
    sample_size: int = 10000,
    seed: int = 2026,
    label_column: str = "label",
    slide_column: str = "source_slide",
    source_id_column: str = "source_id",
) -> pd.DataFrame:
    """Select a fixed, label- and slide-stratified audit subset.

    If the eligible pool has at most ``sample_size`` sources, all sources are
    returned. Otherwise sources are allocated across (label, slide) strata in
    proportion to stratum size using deterministic largest-remainder allocation,
    and within each stratum rows are drawn in a seeded permutation. The result is
    a pure function of the input rows and the seed.
    """
    if sources.empty or len(sources) <= sample_size:
        return sources.sort_values(source_id_column).reset_index(drop=True)
    frame = sources.copy()
    missing = [column for column in (source_id_column, label_column, slide_column) if column not in frame.columns]
    if missing:
        raise KeyError(f"Audit subset input is missing columns: {missing}")
    strata = list(frame.groupby([label_column, slide_column], sort=True, dropna=False))
    sizes = np.array([len(group) for _, group in strata], dtype=np.int64)
    total = int(sizes.sum())
    exact = sizes * sample_size / total
    allocated = exact.astype(np.int64)
    remainder = exact - allocated
    order = np.lexsort((np.arange(len(strata)), -remainder))
    remaining = int(sample_size - allocated.sum())
    for index in order[:remaining]:
        allocated[index] += 1
    rng = np.random.default_rng(seed)
    parts = []
    for (_, group), count in zip(strata, allocated):
        count = int(count)
        if count <= 0:
            continue
        if count >= len(group):
            part = group
        else:
            part = group.iloc[rng.permutation(len(group))[:count]]
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    if len(result) > sample_size:
        result = result.iloc[:sample_size]
    return result.sort_values(source_id_column).reset_index(drop=True)
