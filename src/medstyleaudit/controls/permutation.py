"""Permutation control preserving cue counts."""

from __future__ import annotations

import numpy as np


def permute_assignments(assignments: np.ndarray, seed: int, strata: np.ndarray | None = None) -> np.ndarray:
    result = np.asarray(assignments).copy()
    rng = np.random.default_rng(seed)
    if strata is None:
        return rng.permutation(result)
    strata_array = np.asarray(strata)
    for value in np.unique(strata_array):
        indices = np.flatnonzero(strata_array == value)
        result[indices] = rng.permutation(result[indices])
    return result
