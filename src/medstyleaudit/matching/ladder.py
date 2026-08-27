"""Identification-ladder donor policies with explicit estimand labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .candidate_bank import CandidateBank
from .distance import euclidean_to


def advanced_level_status(level: int, available_columns, required_columns) -> dict[str, object]:
    """Return an honest structured status for unimplemented ladder levels 5/6."""
    if level not in {5, 6}:
        raise ValueError("advanced_level_status is only valid for levels 5 and 6")
    missing = sorted(set(required_columns) - set(available_columns))
    if missing:
        return {"level": level, "status": "unavailable", "reason": f"missing validated columns: {missing}", "missing_columns": missing}
    policy = "lesion-aware" if level == 5 else "appearance-aware"
    return {"level": level, "status": "not_implemented", "reason": f"{policy} matching policy is not implemented"}


def independent_nearest_pairs(bank: CandidateBank, sources: pd.DataFrame, target_hospital: int, k: int = 1) -> pd.DataFrame:
    """Level 3: independently choose nearest within and cross donors."""
    z_columns = [f"__z_{column}" for column in bank.descriptor_columns]
    rows = []
    for _, source in sources.iterrows():
        excluded = {str(source["physical_id"])}; donor_split = "train" if source["split"] in {"val", "test"} else source["split"]
        within = bank.candidates(hospital=int(source["hospital_id"]), label=int(source["label"]), excluded_physical_ids=excluded, split=source["split"], source_ids={source["source_id"]})
        cross = bank.candidates(hospital=int(target_hospital), label=int(source["label"]), excluded_physical_ids=excluded, split=donor_split)
        vector = source[z_columns].to_numpy(float)
        for frame in (within, cross): frame["__distance"] = euclidean_to(vector, frame[z_columns].to_numpy(float))
        within, cross = within.nsmallest(k, "__distance"), cross.nsmallest(k, "__distance")
        if len(within) < k or len(cross) < k: continue
        for index in range(k):
            rows.append({"source_id": source["source_id"], "source_split": source["split"], "source_hospital": int(source["hospital_id"]), "target_hospital": int(target_hospital), "label": int(source["label"]), "within_donor": within.iloc[index]["source_id"], "cross_donor": cross.iloc[index]["source_id"], "matching_distance_within": within.iloc[index]["__distance"], "matching_distance_cross": cross.iloc[index]["__distance"], "donor_index": index})
    return pd.DataFrame(rows)


def random_pairs(bank: CandidateBank, sources: pd.DataFrame, target_hospital: int, k: int = 1, seed: int = 42) -> pd.DataFrame:
    """Level 2: paired random same-label donors with physical exclusion."""
    rng = np.random.default_rng(seed); rows = []
    for _, source in sources.iterrows():
        excluded = {str(source["physical_id"])}; donor_split = "train" if source["split"] in {"val", "test"} else source["split"]
        within = bank.candidates(hospital=int(source["hospital_id"]), label=int(source["label"]), excluded_physical_ids=excluded, split=source["split"], source_ids={source["source_id"]})
        cross = bank.candidates(hospital=int(target_hospital), label=int(source["label"]), excluded_physical_ids=excluded, split=donor_split)
        if len(within) < k or len(cross) < k: continue
        within_indices, cross_indices = rng.choice(len(within), k, False), rng.choice(len(cross), k, False)
        for index, (wi, ci) in enumerate(zip(within_indices, cross_indices)):
            rows.append({"source_id": source["source_id"], "source_split": source["split"], "source_hospital": int(source["hospital_id"]), "target_hospital": int(target_hospital), "label": int(source["label"]), "within_donor": within.iloc[wi]["source_id"], "cross_donor": cross.iloc[ci]["source_id"], "donor_index": index})
    return pd.DataFrame(rows)
