"""Immutable, split-aware donor candidate banks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .distance import Standardizer


@dataclass
class CandidateBank:
    frame: pd.DataFrame
    standardizer: Standardizer
    descriptor_columns: list[str]
    _group_indices: dict[tuple[int, int, str], np.ndarray] = field(default_factory=dict, repr=False)
    _group_trees: dict[tuple[int, int, str], cKDTree] = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls,
        frame: pd.DataFrame,
        descriptor_columns: list[str],
        training_splits: Iterable[str] = ("train",),
    ) -> "CandidateBank":
        required = {"source_id", "split", "hospital_id", "label", "slide_id", "physical_id", *descriptor_columns}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Descriptor table is missing columns: {missing}")
        train = frame[frame["split"].isin(training_splits)]
        if train.empty:
            raise ValueError("Training partition is empty; cannot fit descriptor standardization")
        standardizer = Standardizer.fit(train, descriptor_columns)
        result = frame.copy().reset_index(drop=True)
        standardized = standardizer.transform(result)
        for index, column in enumerate(descriptor_columns):
            result[f"__z_{column}"] = standardized[:, index]
        group_indices: dict[tuple[int, int, str], np.ndarray] = {}
        group_trees: dict[tuple[int, int, str], cKDTree] = {}
        for values, positions in result.groupby(["hospital_id", "label", "split"], sort=False).indices.items():
            key = (int(values[0]), int(values[1]), str(values[2]))
            ordered_positions = np.sort(np.asarray(positions, dtype=np.int64))
            group_indices[key] = ordered_positions
            group_trees[key] = cKDTree(standardized[ordered_positions])
        return cls(result, standardizer, descriptor_columns, group_indices, group_trees)

    @staticmethod
    def _key(hospital: int, label: int, split: str) -> tuple[int, int, str]:
        return int(hospital), int(label), str(split)

    def group_size(self, hospital: int, label: int, split: str) -> int:
        return len(self._group_indices.get(self._key(hospital, label, split), ()))

    def query_group(self, hospital: int, label: int, split: str, vector: np.ndarray, k: int) -> np.ndarray:
        """Return global frame positions of the k nearest indexed candidates."""
        key = self._key(hospital, label, split)
        positions = self._group_indices.get(key)
        if positions is None or not len(positions) or k <= 0:
            return np.empty(0, dtype=np.int64)
        count = min(int(k), len(positions))
        _, local = self._group_trees[key].query(np.asarray(vector, dtype=np.float64), k=count, workers=1)
        return positions[np.atleast_1d(local).astype(np.int64)]

    def query_group_batch(
        self,
        hospital: int,
        label: int,
        split: str,
        vectors: np.ndarray,
        k: int,
        workers: int = 1,
    ) -> np.ndarray:
        """Batch nearest-neighbor queries while retaining global frame positions."""
        key = self._key(hospital, label, split)
        positions = self._group_indices.get(key)
        vectors = np.asarray(vectors, dtype=np.float64)
        if positions is None or not len(positions) or k <= 0:
            return np.empty((len(vectors), 0), dtype=np.int64)
        count = min(int(k), len(positions))
        _, local = self._group_trees[key].query(vectors, k=count, workers=int(workers))
        local = np.asarray(local, dtype=np.int64)
        if count == 1:
            local = local[:, None]
        return positions[local]

    def query_group_radius(self, hospital: int, label: int, split: str, vector: np.ndarray, radius: float) -> np.ndarray:
        """Return every indexed candidate within radius, including distance ties."""
        key = self._key(hospital, label, split)
        positions = self._group_indices.get(key)
        if positions is None or not len(positions):
            return np.empty(0, dtype=np.int64)
        local = self._group_trees[key].query_ball_point(np.asarray(vector, dtype=np.float64), float(radius), workers=1)
        return positions[np.asarray(local, dtype=np.int64)]

    def candidates(
        self,
        *,
        hospital: int,
        label: int,
        excluded_physical_ids: set[str],
        split: str | None = None,
        source_ids: set[object] | None = None,
    ) -> pd.DataFrame:
        if split is not None:
            positions = self._group_indices.get(self._key(hospital, label, split), np.empty(0, dtype=np.int64))
            result = self.frame.iloc[positions]
        else:
            result = self.frame[(self.frame["hospital_id"] == hospital) & (self.frame["label"] == label)]
        mask = ~result["physical_id"].astype(str).isin(excluded_physical_ids)
        if source_ids:
            mask &= ~result["source_id"].isin(source_ids)
        return result.loc[mask].copy()
