import pandas as pd

from medstyleaudit.data.metadata import canonicalize_metadata


def test_patient_then_slide_physical_cluster():
    frame = pd.DataFrame({"id": [1, 2], "split": ["train", "train"], "hospital": [0, 0], "slide": ["s1", "s2"], "patient": ["p1", None], "y": [0, 1]})
    result = canonicalize_metadata(frame)
    assert result["physical_id"].tolist() == ["patient:p1", "slide:s2"]
