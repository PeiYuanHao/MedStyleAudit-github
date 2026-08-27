import pandas as pd

from medstyleaudit.data.slide_mapping import build_patch_mapping, map_slides


def test_mapping_never_guesses_absent_or_ambiguous_slides():
    metadata = pd.DataFrame({"slide_id": ["a", "b"], "split": ["train", "val"], "hospital_id": [0, 1]})
    manifest = pd.DataFrame({"wilds_slide_id": ["a", "a"], "camelyon_wsi": ["x", "y"]})
    result = map_slides(metadata, manifest).set_index("wilds_slide_id")
    assert result.loc["a", "mapping_status"] == "ambiguous"
    assert result.loc["b", "mapping_status"] == "unmapped"


def _patch_metadata(**updates):
    row = {"source_id": 7, "label": 1, "split": "val", "hospital_id": 1, "slide_id": "s1", "patient_id": "p1", "x_coord": 10, "y_coord": 20}
    row.update(updates)
    return pd.DataFrame([row])


def _build(tmp_path, metadata=None, *, slide_status="mapped", wsi_exists=True, annotation_exists=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    wsi = tmp_path / "patient_1_node_0.tif"
    if wsi_exists: wsi.touch()
    annotation_root = tmp_path / "annotations"; annotation_root.mkdir(parents=True)
    if annotation_exists: (annotation_root / "patient_1_node_0.xml").write_text("<Annotations/>", encoding="utf-8")
    slides = pd.DataFrame([{"wilds_slide_id": "s1", "mapping_status": slide_status, "mapping_reason": "test", "camelyon_wsi": "patient_1_node_0", "wsi_path": str(wsi)}])
    return build_patch_mapping(metadata if metadata is not None else _patch_metadata(), slides, annotation_root=annotation_root, patch_size=96, scale=1.0, coordinate_level=0, orientation="identity", coordinate_reference="top_left")


def test_patch_mapping_exact_and_missing_assets_are_explicit(tmp_path):
    exact = _build(tmp_path / "exact")
    assert exact.loc[0, "mapping_status"] == "mapped"
    assert exact.loc[0, "x"] == 10 and exact.loc[0, "y"] == 20
    missing_wsi = _build(tmp_path / "missing_wsi", wsi_exists=False)
    assert missing_wsi.loc[0, "mapping_status"] == "unavailable" and "WSI" in missing_wsi.loc[0, "mapping_reason"]
    missing_annotation = _build(tmp_path / "missing_xml", annotation_exists=False)
    assert missing_annotation.loc[0, "mapping_status"] == "unavailable" and "annotation" in missing_annotation.loc[0, "mapping_reason"]


def test_patch_mapping_preserves_ambiguity_and_missing_coordinates(tmp_path):
    ambiguous = _build(tmp_path / "ambiguous", slide_status="ambiguous")
    assert ambiguous.loc[0, "mapping_status"] == "ambiguous"
    missing_coordinates = _build(tmp_path / "coords", metadata=_patch_metadata(x_coord=pd.NA))
    assert missing_coordinates.loc[0, "mapping_status"] == "unavailable"
    assert "coordinates" in missing_coordinates.loc[0, "mapping_reason"]
