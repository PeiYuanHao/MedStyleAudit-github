"""Physical cluster utilities."""

from __future__ import annotations

import pandas as pd


def physical_cluster(patient: object, slide: object) -> str:
    if pd.notna(patient) and str(patient) not in {"", "nan", "<NA>"}:
        return f"patient:{patient}"
    if pd.isna(slide):
        raise ValueError("Neither reliable patient nor slide identity is available")
    return f"slide:{slide}"


def cluster_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    return {column: int(frame[column].nunique(dropna=True)) for column in columns if column in frame}
