"""Deterministic greedy solver for Balanced Matched-Triplet selection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .candidate_bank import CandidateBank
from .distance import euclidean_to


@dataclass
class BalancedTripletMatcher:
    bank: CandidateBank
    config: Mapping[str, Any]
    donor_reuse: Counter = field(default_factory=Counter)
    donor_slide_reuse: Counter = field(default_factory=Counter)

    def _eligible_by_cap(self, frame: pd.DataFrame) -> pd.DataFrame:
        reuse_cap = int(self.config.get("donor_reuse_cap", 2**31 - 1))
        slide_cap = int(self.config.get("donor_slide_reuse_cap", 2**31 - 1))
        mask = [
            self.donor_reuse[row.source_id] < reuse_cap and self.donor_slide_reuse[row.slide_id] < slide_cap
            for row in frame.itertuples()
        ]
        return frame.loc[mask]

    def _nearest(self, source_vector: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.assign(__distance=pd.Series(dtype=float))
        columns = [f"__z_{column}" for column in self.bank.descriptor_columns]
        result = frame.copy()
        result["__distance"] = euclidean_to(source_vector, result[columns].to_numpy(float))
        pool = int(self.config.get("candidate_pool_size", 64))
        return result.sort_values(["__distance", "source_id"], kind="mergesort").head(pool)

    def match_source(self, source: pd.Series, target_hospital: int, donor_split: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
        """Select exactly K within/cross donor pairs or return a single exclusion reason."""
        settings = self.config
        k_required = int(settings.get("donors_per_source", 1))
        source_vector = source[[f"__z_{column}" for column in self.bank.descriptor_columns]].to_numpy(float)
        excluded = {str(source["physical_id"])}
        within = self.bank.candidates(
            hospital=int(source["hospital_id"]), label=int(source["label"]),
            excluded_physical_ids=excluded, split=source["split"], source_ids={source["source_id"]},
        )
        cross = self.bank.candidates(
            hospital=int(target_hospital), label=int(source["label"]),
            excluded_physical_ids=excluded, split=donor_split,
        )
        within, cross = self._eligible_by_cap(within), self._eligible_by_cap(cross)
        if within.empty:
            return [], "no_within_candidate"
        if cross.empty:
            return [], "no_cross_candidate"
        within, cross = self._nearest(source_vector, within), self._nearest(source_vector, cross)
        tau_d = float(settings.get("tau_distance", np.inf))
        within, cross = within[within["__distance"] <= tau_d], cross[cross["__distance"] <= tau_d]
        if within.empty or cross.empty:
            return [], "distance_threshold"
        lambda_balance = float(settings.get("lambda_balance", 0.0))
        lambda_pair = float(settings.get("lambda_pair", 0.0))
        tau_balance = float(settings.get("tau_balance", np.inf))
        calipers = settings.get("feature_calipers", {})
        rows: list[tuple[float, str, str, pd.Series, pd.Series, float]] = []
        z_columns = [f"__z_{column}" for column in self.bank.descriptor_columns]
        for _, within_row in within.iterrows():
            for _, cross_row in cross.iterrows():
                if str(within_row["physical_id"]) == str(cross_row["physical_id"]):
                    continue
                d_within, d_cross = float(within_row["__distance"]), float(cross_row["__distance"])
                imbalance = abs(d_within - d_cross)
                if imbalance > tau_balance:
                    continue
                if any(abs(float(within_row[name]) - float(cross_row[name])) > float(limit) for name, limit in calipers.items()):
                    continue
                pair_distance = float(np.linalg.norm(within_row[z_columns].to_numpy(float) - cross_row[z_columns].to_numpy(float)))
                cost = d_within + d_cross + lambda_balance * imbalance + lambda_pair * pair_distance
                rows.append((cost, str(within_row["source_id"]), str(cross_row["source_id"]), within_row, cross_row, pair_distance))
        rows.sort(key=lambda value: (value[0], value[1], value[2]))
        selected, used_within, used_cross = [], set(), set()
        used_within_slides, used_cross_slides = set(), set()
        enforce_within_slide_diversity = within["slide_id"].nunique() >= k_required
        enforce_cross_slide_diversity = cross["slide_id"].nunique() >= k_required
        slide_cap = int(settings.get("donor_slide_reuse_cap", 2**31 - 1))
        for cost, _, _, within_row, cross_row, pair_distance in rows:
            if within_row["source_id"] in used_within or cross_row["source_id"] in used_cross:
                continue
            if enforce_within_slide_diversity and within_row["slide_id"] in used_within_slides:
                continue
            if enforce_cross_slide_diversity and cross_row["slide_id"] in used_cross_slides:
                continue
            if self.donor_slide_reuse[within_row["slide_id"]] + sum(item["within_slide"] == within_row["slide_id"] for item in selected) >= slide_cap:
                continue
            if self.donor_slide_reuse[cross_row["slide_id"]] + sum(item["cross_slide"] == cross_row["slide_id"] for item in selected) >= slide_cap:
                continue
            selected.append({
                "source_id": source["source_id"], "source_split": source["split"],
                "source_hospital": int(source["hospital_id"]), "target_hospital": int(target_hospital),
                "source_slide": source["slide_id"], "source_patient": source.get("patient_id", pd.NA),
                "source_physical_id": source["physical_id"], "label": int(source["label"]),
                "within_donor": within_row["source_id"], "within_slide": within_row["slide_id"],
                "within_physical_id": within_row["physical_id"], "cross_donor": cross_row["source_id"],
                "cross_slide": cross_row["slide_id"], "cross_physical_id": cross_row["physical_id"],
                "matching_distance_within": float(within_row["__distance"]),
                "matching_distance_cross": float(cross_row["__distance"]),
                "matching_distance_pair": pair_distance, "matching_cost": float(cost),
                "donor_index": len(selected),
            })
            used_within.add(within_row["source_id"])
            used_cross.add(cross_row["source_id"])
            used_within_slides.add(within_row["slide_id"])
            used_cross_slides.add(cross_row["slide_id"])
            if len(selected) == k_required:
                break
        if len(selected) != k_required:
            return [], "insufficient_balanced_pairs"
        for record in selected:
            for donor, slide in [(record["within_donor"], record["within_slide"]), (record["cross_donor"], record["cross_slide"])]:
                self.donor_reuse[donor] += 1
                self.donor_slide_reuse[slide] += 1
        return selected, None

    def match(self, sources: pd.DataFrame, target_hospitals: Mapping[str, list[int]], show_progress: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        triplets, ledger = [], []
        ordered = sources.sort_values(["source_split", "source_hospital", "source_id"] if "source_split" in sources else ["split", "hospital_id", "source_id"])
        iterator = ordered.iterrows()
        if show_progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, total=len(ordered), desc="Phase 3 balanced matching", unit="source")
        for _, source in iterator:
            split = source.get("source_split", source["split"])
            source_hospital = int(source.get("source_hospital", source["hospital_id"]))
            for target in target_hospitals.get(str(split), []):
                if int(target) == source_hospital:
                    continue
                donor_split = "train" if split in {"val", "test"} else split
                selected, reason = self.match_source(source, int(target), donor_split)
                accepted = reason is None
                ledger.append({"source_id": source["source_id"], "source_split": split, "source_hospital": source_hospital, "target_hospital": int(target), "coverage_status": "accepted" if accepted else "excluded", "exclusion_reason": reason})
                triplets.extend(selected)
        triplets_frame = pd.DataFrame(triplets)
        if not triplets_frame.empty:
            triplets_frame.insert(0, "triplet_id", [f"t{index:09d}" for index in range(len(triplets_frame))])
        return triplets_frame, pd.DataFrame(ledger)
