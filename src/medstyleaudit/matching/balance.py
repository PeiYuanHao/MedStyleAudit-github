"""Paired matching-balance diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def balance_summary(triplets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in triplets.groupby(["source_split", "source_hospital", "target_hospital"]):
        within = group["matching_distance_within"].to_numpy(float)
        cross = group["matching_distance_cross"].to_numpy(float)
        scale = float(np.std(np.concatenate([within, cross]), ddof=1)) if len(group) > 1 else 0.0
        rows.append({
            "source_split": keys[0], "source_hospital": keys[1], "target_hospital": keys[2],
            "n_triplets": len(group), "within_mean": within.mean(), "cross_mean": cross.mean(),
            "within_median": np.median(within), "cross_median": np.median(cross),
            "mean_paired_difference": np.mean(cross - within),
            "paired_smd": (cross.mean() - within.mean()) / scale if scale > 0 else (0.0 if np.array_equal(within, cross) else np.nan),
            "mean_pair_distance": group["matching_distance_pair"].mean(),
        })
    return pd.DataFrame(rows)


def _paired_smd(within: np.ndarray, cross: np.ndarray) -> tuple[float, float]:
    differences = cross - within
    pooled = np.sqrt((np.var(within, ddof=1) + np.var(cross, ddof=1)) / 2) if len(within) > 1 else 0.0
    mean_difference = float(np.mean(differences))
    if pooled > 0:
        return mean_difference, mean_difference / float(pooled)
    return mean_difference, 0.0 if np.allclose(within, cross) else float("nan")


def feature_balance(triplets: pd.DataFrame, descriptors: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Report prespecified descriptor balance for every directed source/target pair."""
    missing = (set(features) | {"source_id"}) - set(descriptors.columns)
    if missing:
        raise KeyError(f"Descriptor columns are missing: {sorted(missing)}")
    required = {"source_split", "source_hospital", "target_hospital", "within_donor", "cross_donor"}
    if required - set(triplets.columns):
        raise KeyError(f"Triplet columns are missing: {sorted(required - set(triplets.columns))}")
    lookup = descriptors.drop_duplicates("source_id").set_index("source_id")
    rows = []
    for keys, group in triplets.groupby(["source_split", "source_hospital", "target_hospital"], sort=True):
        for feature in features:
            within = lookup.loc[group["within_donor"], feature].to_numpy(dtype=float)
            cross = lookup.loc[group["cross_donor"], feature].to_numpy(dtype=float)
            if not np.isfinite(within).all() or not np.isfinite(cross).all():
                raise ValueError(f"Non-finite descriptor values for {feature} in directed pair {keys}")
            difference, smd = _paired_smd(within, cross)
            rows.append({
                "source_split": keys[0], "source_hospital": int(keys[1]), "target_hospital": int(keys[2]),
                "feature": feature, "n_pairs": len(group), "within_mean": float(np.mean(within)),
                "cross_mean": float(np.mean(cross)), "within_median": float(np.median(within)),
                "cross_median": float(np.median(cross)), "paired_mean_difference": difference,
                "paired_smd": smd,
            })
    return pd.DataFrame(rows)


def donor_reuse_summary(triplets: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
    """Emit donor-patch detail rows followed by distribution summary rows."""
    lookup = descriptors.drop_duplicates("source_id").set_index("source_id")
    events = []
    for kind, donor_col, slide_col in (("within", "within_donor", "within_slide"), ("cross", "cross_donor", "cross_slide")):
        frame = triplets[[donor_col, slide_col]].copy()
        frame.columns = ["donor_source_id", "slide"]
        frame["kind"] = kind
        events.append(frame)
    event = pd.concat(events, ignore_index=True)
    counts = event.pivot_table(index=["donor_source_id", "slide"], columns="kind", aggfunc="size", fill_value=0).reset_index()
    for column in ("within", "cross"):
        if column not in counts:
            counts[column] = 0
    counts = counts.rename(columns={"within": "within_reuse", "cross": "cross_reuse"})
    counts["total_reuse"] = counts["within_reuse"] + counts["cross_reuse"]
    counts["hospital"] = counts["donor_source_id"].map(lookup["hospital_id"])
    counts["label"] = counts["donor_source_id"].map(lookup["label"])
    counts.insert(0, "row_type", "donor")
    summaries = []
    values = counts["total_reuse"].to_numpy(dtype=float)
    for name, value in (("min", np.min(values)), ("median", np.median(values)), ("p90", np.quantile(values, .90)), ("p95", np.quantile(values, .95)), ("p99", np.quantile(values, .99)), ("max", np.max(values))):
        summaries.append({"row_type": "summary", "statistic": name, "total_reuse": float(value)})
    return pd.concat([counts, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def slide_reuse_summary(triplets: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
    """Report donor-slide concentration within each directed pair and globally."""
    lookup = descriptors.drop_duplicates("source_id").set_index("source_id")
    events = []
    for kind, donor_col, slide_col in (("within", "within_donor", "within_slide"), ("cross", "cross_donor", "cross_slide")):
        frame = triplets[["source_split", "source_hospital", "target_hospital", donor_col, slide_col]].copy()
        frame.columns = ["source_split", "source_hospital", "target_hospital", "donor_source_id", "slide"]
        frame["kind"] = kind
        frame["hospital"] = frame["donor_source_id"].map(lookup["hospital_id"])
        events.append(frame)
    event = pd.concat(events, ignore_index=True)
    keys = ["source_split", "source_hospital", "target_hospital", "hospital", "slide"]
    detail = event.pivot_table(index=keys, columns="kind", aggfunc="size", fill_value=0).reset_index()
    for column in ("within", "cross"):
        if column not in detail:
            detail[column] = 0
    detail = detail.rename(columns={"within": "within_uses", "cross": "cross_uses"})
    detail["total_donor_uses"] = detail["within_uses"] + detail["cross_uses"]
    pair_totals = detail.groupby(["source_split", "source_hospital", "target_hospital"])["total_donor_uses"].transform("sum")
    detail["directed_pair_fraction"] = detail["total_donor_uses"] / pair_totals
    detail.insert(0, "row_type", "slide")
    summaries = []
    for pair, group in detail.groupby(["source_split", "source_hospital", "target_hospital"], sort=True):
        uses = group["total_donor_uses"].to_numpy(dtype=float)
        shares = uses / uses.sum()
        summaries.append({
            "row_type": "summary", "source_split": pair[0], "source_hospital": pair[1], "target_hospital": pair[2],
            "unique_donor_slides": int(len(group)), "effective_donor_slide_count": float(1.0 / np.sum(shares ** 2)),
            "maximum_slide_share": float(np.max(shares)), "total_donor_uses": int(uses.sum()),
        })
    return pd.concat([detail, pd.DataFrame(summaries)], ignore_index=True, sort=False)


DONOR_REUSE_COLUMNS = ["donor_id", "donor_hospital", "donor_slide", "donor_label", "within_uses", "cross_uses", "total_uses"]
SLIDE_REUSE_COLUMNS = ["hospital", "slide_id", "within_uses", "cross_uses", "total_uses", "share_of_all_donor_uses"]


def donor_reuse_detail(triplets: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
    """Final-suite donor reuse detail: one row per donor patch with within/cross usage."""
    if triplets.empty:
        return pd.DataFrame(columns=DONOR_REUSE_COLUMNS)
    lookup = descriptors.drop_duplicates("source_id").set_index("source_id")
    events = []
    for kind, donor_col, slide_col in (("within", "within_donor", "within_slide"), ("cross", "cross_donor", "cross_slide")):
        frame = triplets[[donor_col, slide_col]].copy()
        frame.columns = ["donor_id", "donor_slide"]
        frame["kind"] = kind
        events.append(frame)
    event = pd.concat(events, ignore_index=True)
    counts = event.pivot_table(index=["donor_id", "donor_slide"], columns="kind", aggfunc="size", fill_value=0).reset_index()
    for column in ("within", "cross"):
        if column not in counts:
            counts[column] = 0
    counts = counts.rename(columns={"within": "within_uses", "cross": "cross_uses"})
    counts["total_uses"] = counts["within_uses"] + counts["cross_uses"]
    counts["donor_hospital"] = counts["donor_id"].map(lookup["hospital_id"])
    counts["donor_label"] = counts["donor_id"].map(lookup["label"])
    return counts[DONOR_REUSE_COLUMNS].sort_values(["donor_hospital", "donor_id"]).reset_index(drop=True)


def donor_reuse_distribution(triplets: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
    """Final-suite donor reuse distribution summary across all donors."""
    detail = donor_reuse_detail(triplets, descriptors)
    values = detail["total_uses"].to_numpy(dtype=float)
    empty = float("nan") if not len(values) else None
    return pd.DataFrame([{
        "n_unique_donors": int(len(detail)),
        "median": empty if empty is not None else float(np.median(values)),
        "p90": empty if empty is not None else float(np.quantile(values, 0.90)),
        "p95": empty if empty is not None else float(np.quantile(values, 0.95)),
        "p99": empty if empty is not None else float(np.quantile(values, 0.99)),
        "max": empty if empty is not None else float(np.max(values)),
    }])


def slide_reuse_detail(triplets: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
    """Final-suite slide reuse detail: one row per donor slide with within/cross usage."""
    if triplets.empty:
        return pd.DataFrame(columns=SLIDE_REUSE_COLUMNS)
    lookup = descriptors.drop_duplicates("source_id").set_index("source_id")
    events = []
    for kind, donor_col, slide_col in (("within", "within_donor", "within_slide"), ("cross", "cross_donor", "cross_slide")):
        frame = triplets[[donor_col, slide_col]].copy()
        frame.columns = ["donor_id", "slide_id"]
        frame["kind"] = kind
        events.append(frame)
    event = pd.concat(events, ignore_index=True)
    event["hospital"] = event["donor_id"].map(lookup["hospital_id"])
    counts = event.pivot_table(index=["hospital", "slide_id"], columns="kind", aggfunc="size", fill_value=0).reset_index()
    for column in ("within", "cross"):
        if column not in counts:
            counts[column] = 0
    counts = counts.rename(columns={"within": "within_uses", "cross": "cross_uses"})
    counts["total_uses"] = counts["within_uses"] + counts["cross_uses"]
    total = int(counts["total_uses"].sum()) if not counts.empty else 0
    counts["share_of_all_donor_uses"] = counts["total_uses"] / total if total else 0.0
    return counts[SLIDE_REUSE_COLUMNS].sort_values(["hospital", "slide_id"]).reset_index(drop=True)


def slide_reuse_distribution(triplets: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
    """Final-suite slide reuse distribution summary across all slides."""
    detail = slide_reuse_detail(triplets, descriptors)
    values = detail["total_uses"].to_numpy(dtype=float)
    empty = float("nan") if not len(values) else None
    return pd.DataFrame([{
        "n_unique_slides": int(len(detail)),
        "median": empty if empty is not None else float(np.median(values)),
        "p90": empty if empty is not None else float(np.quantile(values, 0.90)),
        "p95": empty if empty is not None else float(np.quantile(values, 0.95)),
        "p99": empty if empty is not None else float(np.quantile(values, 0.99)),
        "max": empty if empty is not None else float(np.max(values)),
        "maximum_slide_share": float("nan") if not len(detail) else float(np.max(detail["share_of_all_donor_uses"])),
    }])
