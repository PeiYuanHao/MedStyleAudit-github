"""Deterministic greedy solver for Balanced Matched-Triplet selection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

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
        donor_ids = frame["source_id"].to_numpy()
        slide_ids = frame["slide_id"].to_numpy()
        mask = np.fromiter(
            (self.donor_reuse[donor] < reuse_cap and self.donor_slide_reuse[slide] < slide_cap for donor, slide in zip(donor_ids, slide_ids)),
            dtype=bool,
            count=len(frame),
        )
        return frame.loc[mask]

    def _nearest(self, source_vector: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.assign(__distance=pd.Series(dtype=float))
        columns = [f"__z_{column}" for column in self.bank.descriptor_columns]
        result = frame.copy()
        result["__distance"] = euclidean_to(source_vector, result[columns].to_numpy(float))
        pool = int(self.config.get("candidate_pool_size", 64))
        return result.sort_values(["__distance", "source_id"], kind="mergesort").head(pool)

    def _nearest_indexed(
        self,
        source_vector: np.ndarray,
        *,
        hospital: int,
        label_value: int,
        split: str,
        excluded_physical_ids: set[str],
        source_ids: set[object] | None = None,
        initial_positions: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Recover the exact nearest eligible pool using a prespecified KD-tree group."""
        total = self.bank.group_size(hospital, label_value, split)
        if total == 0:
            return self.bank.frame.iloc[:0].assign(__distance=pd.Series(dtype=float))
        pool = int(self.config.get("candidate_pool_size", 64))
        probe = min(total, max(pool * 2, 128)) if initial_positions is None else len(initial_positions)

        def eligible(positions: np.ndarray) -> pd.DataFrame:
            frame = self.bank.frame.iloc[positions]
            physical_ids = frame["physical_id"].astype(str).to_numpy()
            if len(excluded_physical_ids) == 1:
                mask = physical_ids != next(iter(excluded_physical_ids))
            else:
                mask = ~np.isin(physical_ids, list(excluded_physical_ids))
            if source_ids:
                candidate_source_ids = frame["source_id"].to_numpy()
                if len(source_ids) == 1:
                    mask &= candidate_source_ids != next(iter(source_ids))
                else:
                    mask &= ~np.isin(candidate_source_ids, list(source_ids))
            result = self._eligible_by_cap(frame.loc[mask])
            if result.empty:
                return result.assign(__distance=pd.Series(dtype=float))
            z_columns = [f"__z_{column}" for column in self.bank.descriptor_columns]
            distances = euclidean_to(source_vector, result[z_columns].to_numpy(float))
            order = np.lexsort((result["source_id"].astype(str).to_numpy(), distances))
            result = result.iloc[order].copy()
            result["__distance"] = distances[order]
            return result

        while True:
            positions = initial_positions if initial_positions is not None else self.bank.query_group(hospital, label_value, split, source_vector, probe)
            initial_positions = None
            result = eligible(positions)
            if len(result) >= pool:
                cutoff = float(result.iloc[pool - 1]["__distance"])
                if probe < total:
                    tolerance = max(1e-12, abs(cutoff) * 1e-12)
                    # The farthest eligible item is a conservative lower bound on
                    # the probe boundary. It avoids a second distance pass in the
                    # common case; a lower bound can only trigger a safe extra tie query.
                    boundary = float(result["__distance"].max())
                    if cutoff + tolerance >= boundary:
                        tied_positions = self.bank.query_group_radius(hospital, label_value, split, source_vector, cutoff + tolerance)
                        result = eligible(tied_positions)
                return result.head(pool)
            if probe >= total:
                return result.head(pool)
            probe = min(total, probe * 2)

    def _balanced_pair_candidates(self, within: pd.DataFrame, cross: pd.DataFrame) -> Iterator[tuple[float, int, int, float]]:
        """Vectorize the fixed 64x64 pair costs while preserving deterministic ordering."""
        settings = self.config
        d_within = within["__distance"].to_numpy(float)
        d_cross = cross["__distance"].to_numpy(float)
        imbalance = np.abs(d_within[:, None] - d_cross[None, :])
        valid = imbalance <= float(settings.get("tau_balance", np.inf))
        within_physical = within["physical_id"].astype(str).to_numpy()
        cross_physical = cross["physical_id"].astype(str).to_numpy()
        valid &= within_physical[:, None] != cross_physical[None, :]
        for name, limit in settings.get("feature_calipers", {}).items():
            valid &= np.abs(within[name].to_numpy(float)[:, None] - cross[name].to_numpy(float)[None, :]) <= float(limit)
        if not valid.any():
            return
        z_columns = [f"__z_{column}" for column in self.bank.descriptor_columns]
        within_z, cross_z = within[z_columns].to_numpy(float), cross[z_columns].to_numpy(float)
        pair_distance = cdist(within_z, cross_z, metric="euclidean")
        cost = (
            d_within[:, None]
            + d_cross[None, :]
            + float(settings.get("lambda_balance", 0.0)) * imbalance
            + float(settings.get("lambda_pair", 0.0)) * pair_distance
        )
        within_indices, cross_indices = np.nonzero(valid)
        within_ids = within["source_id"].astype(str).to_numpy()[within_indices]
        cross_ids = cross["source_id"].astype(str).to_numpy()[cross_indices]
        candidate_costs = cost[within_indices, cross_indices]
        order = np.lexsort((cross_ids, within_ids, candidate_costs))
        for index in order:
            within_index, cross_index = int(within_indices[index]), int(cross_indices[index])
            yield float(candidate_costs[index]), within_index, cross_index, float(pair_distance[within_index, cross_index])

    def match_source(
        self,
        source: pd.Series,
        target_hospital: int,
        donor_split: str | None = None,
        candidate_prefetch: Mapping[tuple[object, int, int, str], np.ndarray] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Select exactly K within/cross donor pairs or return a single exclusion reason."""
        settings = self.config
        k_required = int(settings.get("donors_per_source", 1))
        source_vector = source[[f"__z_{column}" for column in self.bank.descriptor_columns]].to_numpy(float)
        excluded = {str(source["physical_id"])}
        prefetch = candidate_prefetch or {}
        source_id, label_value = source["source_id"], int(source["label"])
        within_key = (source_id, int(source["hospital_id"]), label_value, str(source["split"]))
        cross_key = (source_id, int(target_hospital), label_value, str(donor_split))
        within = self._nearest_indexed(
            source_vector,
            hospital=int(source["hospital_id"]), label_value=label_value,
            excluded_physical_ids=excluded, split=str(source["split"]), source_ids={source["source_id"]},
            initial_positions=prefetch.get(within_key),
        )
        cross = self._nearest_indexed(
            source_vector,
            hospital=int(target_hospital), label_value=label_value,
            excluded_physical_ids=excluded, split=str(donor_split),
            initial_positions=prefetch.get(cross_key),
        )
        if within.empty:
            return [], "no_within_candidate"
        if cross.empty:
            return [], "no_cross_candidate"
        tau_d = float(settings.get("tau_distance", np.inf))
        within, cross = within[within["__distance"] <= tau_d], cross[cross["__distance"] <= tau_d]
        if within.empty or cross.empty:
            return [], "distance_threshold"
        rows = self._balanced_pair_candidates(within, cross)
        selected, used_within, used_cross = [], set(), set()
        used_within_slides, used_cross_slides = set(), set()
        enforce_within_slide_diversity = within["slide_id"].nunique() >= k_required
        enforce_cross_slide_diversity = cross["slide_id"].nunique() >= k_required
        slide_cap = int(settings.get("donor_slide_reuse_cap", 2**31 - 1))
        for cost, within_index, cross_index, pair_distance in rows:
            within_row, cross_row = within.iloc[within_index], cross.iloc[cross_index]
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

    def _prefetch_candidates(self, sources: pd.DataFrame, target_hospitals: Mapping[str, list[int]]) -> dict[tuple[object, int, int, str], np.ndarray]:
        """Batch immutable KD-tree queries; dynamic reuse constraints remain sequential."""
        z_columns = [f"__z_{column}" for column in self.bank.descriptor_columns]
        requests: dict[tuple[int, int, str], list[tuple[object, np.ndarray]]] = {}
        for _, source in sources.iterrows():
            split = str(source.get("source_split", source["split"]))
            hospital = int(source.get("source_hospital", source["hospital_id"]))
            label_value = int(source["label"])
            donor_split = "train" if split in {"val", "test"} else split
            keys = {(hospital, label_value, split)}
            keys.update((int(target), label_value, donor_split) for target in target_hospitals.get(split, []) if int(target) != hospital)
            vector = source[z_columns].to_numpy(dtype=np.float64)
            for key in keys:
                requests.setdefault(key, []).append((source["source_id"], vector))
        prefetch: dict[tuple[object, int, int, str], np.ndarray] = {}
        query_size = max(int(self.config.get("candidate_pool_size", 64)) * 2, 128)
        for (hospital, label_value, split), items in requests.items():
            positions = self.bank.query_group_batch(
                hospital,
                label_value,
                split,
                np.stack([item[1] for item in items]),
                query_size,
                workers=int(self.config.get("query_workers", 1)),
            )
            for (source_id, _), row in zip(items, positions):
                prefetch[(source_id, hospital, label_value, split)] = row
        return prefetch

    def match(self, sources: pd.DataFrame, target_hospitals: Mapping[str, list[int]], show_progress: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        triplets, ledger = [], []
        ordered = sources.sort_values(["source_split", "source_hospital", "source_id"] if "source_split" in sources else ["split", "hospital_id", "source_id"])
        progress = None
        if show_progress:
            from tqdm.auto import tqdm
            progress = tqdm(total=len(ordered), desc="Phase 3 balanced matching", unit="source")
        batch_size = int(self.config.get("query_batch_size", 512))
        for start in range(0, len(ordered), batch_size):
            chunk = ordered.iloc[start : start + batch_size]
            prefetch = self._prefetch_candidates(chunk, target_hospitals)
            for _, source in chunk.iterrows():
                split = source.get("source_split", source["split"])
                source_hospital = int(source.get("source_hospital", source["hospital_id"]))
                for target in target_hospitals.get(str(split), []):
                    if int(target) == source_hospital:
                        continue
                    donor_split = "train" if split in {"val", "test"} else split
                    selected, reason = self.match_source(source, int(target), donor_split, prefetch)
                    accepted = reason is None
                    ledger.append({"source_id": source["source_id"], "source_split": split, "source_hospital": source_hospital, "target_hospital": int(target), "coverage_status": "accepted" if accepted else "excluded", "exclusion_reason": reason})
                    triplets.extend(selected)
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()
        triplets_frame = pd.DataFrame(triplets)
        if not triplets_frame.empty:
            triplets_frame.insert(0, "triplet_id", [f"t{index:09d}" for index in range(len(triplets_frame))])
        return triplets_frame, pd.DataFrame(ledger)
