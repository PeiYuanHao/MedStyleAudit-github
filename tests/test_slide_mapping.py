import pandas as pd

from medstyleaudit.data.slide_mapping import map_slides


def test_mapping_never_guesses_absent_or_ambiguous_slides():
    metadata = pd.DataFrame({"slide_id": ["a", "b"], "split": ["train", "val"], "hospital_id": [0, 1]})
    manifest = pd.DataFrame({"wilds_slide_id": ["a", "a"], "camelyon_wsi": ["x", "y"]})
    result = map_slides(metadata, manifest).set_index("wilds_slide_id")
    assert result.loc["a", "mapping_status"] == "ambiguous"
    assert result.loc["b", "mapping_status"] == "unmapped"
