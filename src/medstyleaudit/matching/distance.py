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
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> "Standardizer":
        values = frame[columns].to_numpy(dtype=np.float64)
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0, ddof=0)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        return cls(tuple(columns), mean, scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[list(self.columns)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Matching descriptors contain non-finite values")
        return (values - self.mean) / self.scale

    def to_dict(self) -> dict[str, object]:
        return {"columns": list(self.columns), "mean": self.mean.tolist(), "scale": self.scale.tolist()}


def euclidean_to(source: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return np.linalg.norm(candidates - source[None, :], axis=1)
