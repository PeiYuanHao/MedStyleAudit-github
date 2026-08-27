"""Normalized-logit and source-level paired audit metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_logit_normalizer(logits: np.ndarray, q_min: float = 1e-3) -> dict[str, float]:
    values = np.asarray(logits, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("ID-validation logits must be a non-empty finite array")
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return {"median": float(np.median(values)), "scale": float(max(q3 - q1, q_min)), "q_min": float(q_min)}


def normalize_logits(values: np.ndarray | pd.Series, normalizer: dict[str, float]) -> np.ndarray:
    return (np.asarray(values, dtype=float) - normalizer["median"]) / normalizer["scale"]


def donor_pair_metrics(predictions: pd.DataFrame, normalizer: dict[str, float]) -> pd.DataFrame:
    """Retain every donor pair while adding normalized paired changes."""
    required = {"source_id", "source_split", "source_hospital", "target_hospital", "original_logit", "within_logit", "cross_logit"}
    if not required.issubset(predictions):
        raise ValueError(f"Prediction table missing columns: {sorted(required - set(predictions))}")
    frame = predictions.copy()
    for column in ["original_logit", "within_logit", "cross_logit"]:
        frame[f"normalized_{column}"] = normalize_logits(frame[column], normalizer)
    frame["absolute_within_change"] = abs(frame["normalized_within_logit"] - frame["normalized_original_logit"])
    frame["absolute_cross_change"] = abs(frame["normalized_cross_logit"] - frame["normalized_original_logit"])
    frame["HCS_pair"] = frame["absolute_cross_change"] - frame["absolute_within_change"]
    frame["within_signed_shift_pair"] = frame["normalized_within_logit"] - frame["normalized_original_logit"]
    frame["cross_signed_shift_pair"] = frame["normalized_cross_logit"] - frame["normalized_original_logit"]
    frame["HCE_pair"] = frame["normalized_cross_logit"] - frame["normalized_within_logit"]
    return frame


def source_metrics(predictions: pd.DataFrame, normalizer: dict[str, float]) -> pd.DataFrame:
    """Collapse multiplicity-preserving donor rows into one paired source record."""
    frame = donor_pair_metrics(predictions, normalizer)
    group_columns = ["source_id", "source_split", "source_hospital", "target_hospital"]
    passthrough = [column for column in ["source_slide", "source_patient", "source_physical_id", "label", "seed", "backbone", "common_support"] if column in frame]
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        original_values = group["normalized_original_logit"].to_numpy(float)
        if not np.allclose(original_values, original_values[0], rtol=0, atol=0):
            raise ValueError(f"Original logit varies within source/pair group: {keys}")
        delta_within = float(group["absolute_within_change"].mean())
        delta_cross = float(group["absolute_cross_change"].mean())
        mean_within = float(group["normalized_within_logit"].mean())
        mean_cross = float(group["normalized_cross_logit"].mean())
        record = dict(zip(group_columns, keys))
        record.update({
            "n_donor_pairs": len(group), "normalized_original_logit": original_values[0],
            "delta_within": delta_within, "delta_cross": delta_cross,
            "HCS_source": delta_cross - delta_within,
            "within_signed_shift": mean_within - original_values[0],
            "cross_signed_shift": mean_cross - original_values[0],
            "HCE_source": mean_cross - mean_within,
        })
        record.update({column: group.iloc[0][column] for column in passthrough})
        rows.append(record)
    return pd.DataFrame(rows)
