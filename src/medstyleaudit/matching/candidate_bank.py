"""Immutable, split-aware donor candidate banks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .distance import Standardizer


@dataclass
class CandidateBank:
    frame: pd.DataFrame
    standardizer: Standardizer
    descriptor_columns: list[str]

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
        return cls(result, standardizer, descriptor_columns)

    def candidates(
        self,
        *,
        hospital: int,
        label: int,
        excluded_physical_ids: set[str],
        split: str | None = None,
        source_ids: set[object] | None = None,
    ) -> pd.DataFrame:
        mask = (self.frame["hospital_id"] == hospital) & (self.frame["label"] == label)
        if split is not None:
            mask &= self.frame["split"] == split
        mask &= ~self.frame["physical_id"].astype(str).isin(excluded_physical_ids)
        if source_ids:
            mask &= ~self.frame["source_id"].isin(source_ids)
        return self.frame.loc[mask].copy()
