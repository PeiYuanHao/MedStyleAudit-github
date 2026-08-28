"""Fixed-triplet crossed multiplier bootstrap."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def crossed_multiplier_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    cluster_columns: Sequence[str],
    draws: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap a mean using product exponential weights across physical roles."""
    if frame.empty:
        return {"status": "unavailable", "n": 0}
    values = frame[value_column].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    identities = {column: frame[column].astype(str).to_numpy() for column in cluster_columns if column in frame}
    all_identities = np.unique(np.concatenate(list(identities.values()))) if identities else np.asarray([], dtype=str)
    for draw in range(draws):
        weights = np.ones(len(frame), dtype=float)
        if identities:
            sampled = rng.exponential(1.0, size=len(all_identities))
            lookup = dict(zip(all_identities, sampled))
        for column, ids in identities.items():
            weights *= np.asarray([lookup[item] for item in ids])
        estimates[draw] = np.average(values, weights=weights)
    alpha = 1 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return {"status": "available", "n": len(frame), "draws": draws, "estimate": float(values.mean()), "ci_lower": float(lower), "ci_upper": float(upper), "standard_error": float(estimates.std(ddof=1)), "confidence": confidence}


def aggregate_seed_estimates(point_estimates: Sequence[float]) -> dict[str, float | int]:
    """Aggregate per-seed point estimates with mean/std without resampling seeds.

    The reduced final suite has exactly three seeds. Each seed is bootstrapped
    independently over physical source/donor identities; the three seed-level
    point estimates are then combined only with a plain mean and standard
    deviation, never by treating the seeds as a large sampling distribution.
    """
    values = np.asarray(point_estimates, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        return {"n_seeds": int(values.size), "mean": float("nan"), "std": float("nan")}
    return {
        "n_seeds": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }
