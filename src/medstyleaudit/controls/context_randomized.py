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


def independent_context_assignments(
    sources: pd.DataFrame,
    donor_bank: pd.DataFrame,
    *,
    seed: int,
    donor_reuse_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign realistic donor contexts without conditioning on source labels."""
    required = {"source_id", "label", "hospital_id", "physical_id", "slide_id"}
    if required - set(sources) or required - set(donor_bank):
        raise KeyError(f"Assignment metadata requires {sorted(required)}")
    if len(sources) > len(donor_bank) * donor_reuse_cap:
        raise ValueError("Context-randomized donor reuse capacity is insufficient")
    rng = np.random.default_rng(seed)
    expanded = np.repeat(np.arange(len(donor_bank)), donor_reuse_cap)
    rng.shuffle(expanded)
    rows, cursor = [], 0
    donors = donor_bank.reset_index(drop=True)
    for source in sources.sort_values("source_id", kind="mergesort").itertuples(index=False):
        chosen = None
        for offset in range(cursor, len(expanded)):
            donor = donors.iloc[int(expanded[offset])]
            if donor.source_id == source.source_id or str(donor.physical_id) == str(source.physical_id):
                continue
            chosen = donor
            expanded[cursor], expanded[offset] = expanded[offset], expanded[cursor]
            cursor += 1
            break
        if chosen is None:
            raise ValueError(f"No independent context donor remained for source {source.source_id}")
        rows.append({
            "source_id": source.source_id, "source_label": int(source.label), "source_hospital": int(source.hospital_id),
            "donor_source_id": chosen.source_id, "donor_label": int(chosen.label), "donor_hospital": int(chosen.hospital_id),
            "donor_slide": chosen.slide_id, "donor_physical_id": chosen.physical_id,
        })
    ledger = pd.DataFrame(rows)
    reuse = ledger["donor_source_id"].value_counts()
    diagnostics = []
    for label, group in ledger.groupby("source_label"):
        diagnostics.append({"source_label": int(label), "n": len(group), "donor_positive_rate": float(group["donor_label"].mean()), "unique_donors": int(group["donor_source_id"].nunique()), "max_donor_reuse": int(reuse.max())})
    return ledger, pd.DataFrame(diagnostics)
