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
        rows.append({
            "camelyon_wsi": path.stem,
            "wsi_path": str(path.resolve()),
            "patient_id": match.group("patient") if match else pd.NA,
            "node": match.group("node") if match else pd.NA,
        })
    return pd.DataFrame(rows)


def exact_manifest_from_metadata(metadata: pd.DataFrame, wsi_root: str | Path) -> pd.DataFrame:
    """Derive an exact manifest from unique WILDS patient/node fields and WSI stems."""
    if not {"slide_id", "patient_id", "node"}.issubset(metadata.columns):
        raise ValueError("Exact automatic mapping requires slide_id, patient_id, and node metadata")
    discovered = discover_wsi_files(wsi_root)
    if discovered.empty:
        return pd.DataFrame(columns=["wilds_slide_id", "camelyon_wsi", "patient_id", "wsi_path"])
    rows = []
    for slide_id, group in metadata.groupby("slide_id"):
        identities = group[["patient_id", "node"]].drop_duplicates()
        if len(identities) != 1:
            continue
        patient, node = identities.iloc[0]
        key = re.sub(r"[^a-z0-9]", "", f"patient_{patient}_node_{node}".lower())
        matches = discovered[
            discovered["camelyon_wsi"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True) == key
        ]
        for match in matches.itertuples():
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
            "annotation_path": record.get("annotation_path", pd.NA),
        })
    return pd.DataFrame(rows)


def _present(value: object) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _annotation_candidates(root: str | Path | None) -> dict[str, list[Path]]:
    if root is None or not Path(root).is_dir():
        return {}
    result: dict[str, list[Path]] = {}
    for path in sorted(Path(root).rglob("*.xml")):
        result.setdefault(path.stem.casefold(), []).append(path.resolve())
    return result


def build_patch_mapping(
    metadata: pd.DataFrame,
    slide_mapping: pd.DataFrame,
    *,
    annotation_root: str | Path | None,
    patch_size: int | None,
    scale: float | None,
    coordinate_level: int | None,
    orientation: str | None,
    coordinate_reference: str | None,
) -> pd.DataFrame:
    """Build an explicit patch-to-WSI/XML ledger without inferred identities or coordinates."""
    coordinate_columns = ("x", "y") if {"x", "y"}.issubset(metadata) else ("x_coord", "y_coord")
    slide_lookup = slide_mapping.set_index("wilds_slide_id", drop=False) if "wilds_slide_id" in slide_mapping else pd.DataFrame()
    annotations = _annotation_candidates(annotation_root)
    rows = []
    for record in metadata.to_dict("records"):
        slide_id = record.get("slide_id")
        matches = pd.DataFrame()
        if not slide_lookup.empty:
            matches = slide_mapping[slide_mapping["wilds_slide_id"].astype(str) == str(slide_id)]
        slide = matches.iloc[0].to_dict() if len(matches) == 1 else {}
        slide_status = slide.get("mapping_status", "unavailable")
        wsi_path = slide.get("wsi_path", pd.NA)
        camelyon_wsi = slide.get("camelyon_wsi", pd.NA)
        annotation_path = slide.get("annotation_path", pd.NA)
        status, reason = "mapped", "exact slide, coordinate, WSI, and annotation mapping"
        if len(matches) > 1 or slide_status == "ambiguous":
            status, reason = "ambiguous", "ambiguous WSI mapping"
        elif slide_status != "mapped":
            status, reason = "unavailable", str(slide.get("mapping_reason", "no exact WSI mapping"))
        elif not all(_present(record.get(column)) for column in coordinate_columns):
            status, reason = "unavailable", "missing exact patch coordinates"
        elif patch_size is None or int(patch_size) <= 0 or scale is None or float(scale) <= 0:
            status, reason = "unavailable", "patch_size and scale must be explicitly configured"
        elif coordinate_level is None:
            status, reason = "unavailable", "coordinate pyramid level is not configured"
        elif orientation != "identity":
            status, reason = "unavailable", "WSI orientation must be explicitly validated as identity"
        elif coordinate_reference != "top_left":
            status, reason = "unavailable", "patch coordinate reference must be explicitly configured as top_left"
        elif not _present(wsi_path) or not Path(str(wsi_path)).is_file():
            status, reason = "unavailable", "mapped WSI file is missing"
        else:
            if not _present(annotation_path) and _present(camelyon_wsi):
                candidates = annotations.get(str(camelyon_wsi).casefold(), [])
                if len(candidates) == 1:
                    annotation_path = str(candidates[0])
                elif len(candidates) > 1:
                    status, reason = "ambiguous", "multiple exact annotation XML matches"
                else:
                    status, reason = "unavailable", "exact annotation XML is missing"
            if status == "mapped" and (not _present(annotation_path) or not Path(str(annotation_path)).is_file()):
                status, reason = "unavailable", "mapped annotation XML file is missing"
        rows.append({
            "source_id": record.get("source_id"),
            "label": record.get("label"),
            "split": record.get("split"),
            "hospital_id": record.get("hospital_id"),
            "slide_id": slide_id,
            "patient_id": record.get("patient_id", slide.get("patient_id", pd.NA)),
            "camelyon_wsi": camelyon_wsi,
            "wsi_path": wsi_path,
            "x": record.get(coordinate_columns[0], pd.NA),
            "y": record.get(coordinate_columns[1], pd.NA),
            "patch_size": patch_size,
            "scale": scale,
            "coordinate_level": coordinate_level,
            "orientation": orientation,
            "coordinate_reference": coordinate_reference,
            "annotation_path": annotation_path,
            "mapping_status": status,
            "mapping_reason": reason,
        })
    return pd.DataFrame(rows)
