"""Explicit WILDS-to-CAMELYON17 slide mapping."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def discover_wsi_files(root: str | Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"patient[_-]?(?P<patient>\d+)[_-]node[_-]?(?P<node>\d+)", re.I)
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in {".tif", ".tiff", ".svs"}:
            continue
        match = pattern.search(path.stem)
        rows.append({"camelyon_wsi": path.stem, "wsi_path": str(path.resolve()), "patient_id": match.group("patient") if match else pd.NA})
    return pd.DataFrame(rows)


def exact_manifest_from_metadata(metadata: pd.DataFrame, wsi_root: str | Path) -> pd.DataFrame:
    """Derive an exact manifest from unique WILDS patient/node fields and WSI stems."""
    if not {"slide_id", "patient_id", "node"}.issubset(metadata.columns):
        raise ValueError("Exact automatic mapping requires slide_id, patient_id, and node metadata")
    discovered = discover_wsi_files(wsi_root)
    if discovered.empty:
        return pd.DataFrame(columns=["wilds_slide_id", "camelyon_wsi", "patient_id", "wsi_path"])
    normalized = {re.sub(r"[^a-z0-9]", "", row.camelyon_wsi.lower()): row for row in discovered.itertuples()}
    rows = []
    for slide_id, group in metadata.groupby("slide_id"):
        identities = group[["patient_id", "node"]].drop_duplicates()
        if len(identities) != 1:
            continue
        patient, node = identities.iloc[0]
        key = re.sub(r"[^a-z0-9]", "", f"patient_{patient}_node_{node}".lower())
        if key in normalized:
            match = normalized[key]
            rows.append({"wilds_slide_id": slide_id, "camelyon_wsi": match.camelyon_wsi, "patient_id": patient, "wsi_path": match.wsi_path})
    return pd.DataFrame(rows)


def map_slides(metadata: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Map only exact normalized names; ambiguous and absent matches remain explicit."""
    if "wilds_slide_id" not in manifest and "slide_id" in manifest:
        manifest = manifest.rename(columns={"slide_id": "wilds_slide_id"})
    required = {"wilds_slide_id", "camelyon_wsi"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Slide manifest requires columns: {sorted(required)}")
    duplicate_ids = set(manifest.loc[manifest["wilds_slide_id"].duplicated(False), "wilds_slide_id"].astype(str))
    lookup = manifest.drop_duplicates("wilds_slide_id", keep=False).set_index("wilds_slide_id")
    rows = []
    for slide_id, group in metadata.groupby("slide_id", sort=True):
        key = str(slide_id)
        if key in duplicate_ids:
            status, reason, record = "ambiguous", "multiple manifest rows", {}
        elif slide_id in lookup.index or key in lookup.index:
            index = slide_id if slide_id in lookup.index else key
            record = lookup.loc[index].to_dict()
            status, reason = "mapped", "exact manifest mapping"
        else:
            status, reason, record = "unmapped", "no exact manifest row", {}
        rows.append({
            "wilds_slide_id": slide_id,
            "wilds_split": ";".join(sorted(group["split"].astype(str).unique())),
            "hospital_id": ";".join(sorted(group["hospital_id"].astype(str).unique())),
            "camelyon_wsi": record.get("camelyon_wsi", pd.NA),
            "patient_id": record.get("patient_id", pd.NA),
            "mapping_status": status,
            "mapping_reason": reason,
            "wsi_path": record.get("wsi_path", pd.NA),
        })
    return pd.DataFrame(rows)
