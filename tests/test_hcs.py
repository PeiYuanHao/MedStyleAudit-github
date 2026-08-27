import numpy as np
import pandas as pd

from medstyleaudit.audit.hcs import directed_hcs
from medstyleaudit.audit.metrics import fit_logit_normalizer, source_metrics


def test_hcs_known_answer_and_negative_values_retained():
    predictions = pd.DataFrame({
        "source_id": [1, 1, 2], "source_split": ["val"] * 3, "source_hospital": [1] * 3, "target_hospital": [0] * 3,
        "original_logit": [0, 0, 2], "within_logit": [1, -1, 4], "cross_logit": [2, -2, 3],
    })
    normalizer = {"median": 0, "scale": 1, "q_min": .001}
    sources = source_metrics(predictions, normalizer).sort_values("source_id")
    assert np.allclose(sources["HCS_source"], [1, -1])
    assert directed_hcs(sources)["hcs"].iloc[0] == 0


def test_logit_normalizer_uses_iqr_floor():
    result = fit_logit_normalizer(np.ones(4), q_min=.25)
    assert result == {"median": 1.0, "scale": .25, "q_min": .25}
