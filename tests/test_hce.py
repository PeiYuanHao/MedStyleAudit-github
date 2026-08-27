import pandas as pd

from medstyleaudit.audit.metrics import source_metrics


def test_hce_is_cross_minus_within_and_not_clipped():
    predictions = pd.DataFrame({"source_id": [1, 1], "source_split": ["val"] * 2, "source_hospital": [1] * 2, "target_hospital": [0] * 2, "original_logit": [10, 10], "within_logit": [12, 14], "cross_logit": [8, 10]})
    result = source_metrics(predictions, {"median": 0, "scale": 2, "q_min": .001})
    assert result["HCE_source"].iloc[0] == -2
