"""Descriptor standardization and distances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Standardizer:
    columns: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> Standardizer:
        values = frame[columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Training matching descriptors contain non-finite values")
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0, ddof=0)
        return cls(tuple(columns), mean, scale)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Standardizer:
        columns = tuple(str(column) for column in payload["columns"])
        mean = np.asarray(payload["mean"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        if len(columns) != len(mean) or len(columns) != len(scale):
            raise ValueError("Serialized standardizer columns, means, and scales must have equal length")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale < 0).any():
            raise ValueError("Serialized standardizer contains an invalid training-bank scale")
        return cls(columns, mean, scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[list(self.columns)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Matching descriptors contain non-finite values")
        # A constant training feature has no effect on Euclidean matching.  Use a
        # safe transform denominator while retaining its true fitted scale (zero)
        # for the manuscript-defined paired SMD diagnostic.
        denominator = np.where(self.scale == 0, 1.0, self.scale)
        return (values - self.mean) / denominator

    def to_dict(self) -> dict[str, object]:
        return {"columns": list(self.columns), "mean": self.mean.tolist(), "scale": self.scale.tolist()}


def euclidean_to(source: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return np.linalg.norm(candidates - source[None, :], axis=1)
