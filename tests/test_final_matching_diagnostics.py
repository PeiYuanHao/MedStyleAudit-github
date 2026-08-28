import numpy as np
import pandas as pd

from medstyleaudit.matching.balance import donor_reuse_summary, feature_balance, slide_reuse_summary


def _inputs():
    descriptors = pd.DataFrame({
        "source_id": [10, 11, 20, 21], "hospital_id": [1, 1, 0, 0], "slide_id": ["a", "b", "c", "d"],
        "label": [0, 0, 0, 0], "f1": [1.0, 2.0, 1.1, 2.1], "f2": [4.0, 6.0, 4.0, 6.0],
    })
    triplets = pd.DataFrame({
        "source_split": ["val", "val"], "source_hospital": [1, 1], "target_hospital": [0, 0],
        "within_donor": [10, 11], "within_slide": ["a", "b"], "cross_donor": [20, 21], "cross_slide": ["c", "d"],
    })
    return triplets, descriptors


def test_feature_balance_reports_every_primary_feature():
    triplets, descriptors = _inputs()
    result = feature_balance(triplets, descriptors, ["f1", "f2"])
    assert set(result["feature"]) == {"f1", "f2"}
    assert result.set_index("feature").loc["f2", "paired_smd"] == 0
    assert np.isfinite(result["paired_smd"]).all()


def test_donor_reuse_includes_detail_and_required_quantiles():
    triplets, descriptors = _inputs()
    result = donor_reuse_summary(pd.concat([triplets, triplets]), descriptors)
    assert result[result["row_type"] == "donor"]["total_reuse"].max() == 2
    assert set(result[result["row_type"] == "summary"]["statistic"]) == {"min", "median", "p90", "p95", "p99", "max"}


def test_slide_reuse_reports_effective_count_and_maximum_share():
    triplets, descriptors = _inputs()
    result = slide_reuse_summary(triplets, descriptors)
    summary = result[result["row_type"] == "summary"].iloc[0]
    assert summary.unique_donor_slides == 4
    assert summary.effective_donor_slide_count == 4
    assert summary.maximum_slide_share == 0.25
