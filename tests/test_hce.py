import pandas as pd

from medstyleaudit.audit.hce import pair_weighted_hce
from medstyleaudit.audit.metrics import source_metrics


def test_hce_is_cross_minus_within_and_not_clipped():
    predictions = pd.DataFrame({"source_id": [1, 1], "source_split": ["val"] * 2, "source_hospital": [1] * 2, "target_hospital": [0] * 2, "original_logit": [10, 10], "within_logit": [12, 14], "cross_logit": [8, 10]})
    result = source_metrics(predictions, {"median": 0, "scale": 2, "q_min": .001})
    assert result["HCE_source"].iloc[0] == -2


def test_pair_weighted_hce_has_same_completeness_semantics_as_hcs():
    directed = pd.DataFrame({"source_split": ["val"], "source_hospital": [1], "target_hospital": [0], "hce": [-2.0]})
    result = pair_weighted_hce(directed, {(1, 0), (1, 3)})
    assert result.loc[0, "status"] == "unavailable"
    assert pd.isna(result.loc[0, "hce_pair_weighted"])
    assert result.loc[0, "missing_pairs"] == [(1, 3)]
