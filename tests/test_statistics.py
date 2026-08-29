import numpy as np
import pandas as pd

from medstyleaudit.statistics.bootstrap import crossed_multiplier_bootstrap, crossed_multiplier_grouped_bootstrap
from medstyleaudit.statistics.multiple_testing import benjamini_hochberg


def test_crossed_bootstrap_is_reproducible():
    frame = pd.DataFrame({"value": [1., 2., 3.], "source": ["a", "b", "c"], "donor": ["x", "x", "y"]})
    first = crossed_multiplier_bootstrap(frame, "value", ["source", "donor"], draws=100, seed=9)
    second = crossed_multiplier_bootstrap(frame, "value", ["source", "donor"], draws=100, seed=9)
    assert first == second


def test_grouped_bootstrap_preserves_equal_directed_pair_weighting():
    frame = pd.DataFrame({
        "value": [0.0, 2.0, 10.0],
        "source": ["a", "b", "c"],
        "within": ["w1", "w2", "w3"],
        "cross": ["x1", "x2", "x3"],
        "source_hospital": [1, 1, 1],
        "target_hospital": [0, 0, 3],
    })
    result = crossed_multiplier_grouped_bootstrap(
        frame,
        "value",
        ["source", "within", "cross"],
        ["source_hospital", "target_hospital"],
        draws=100,
        seed=9,
    )
    assert result["estimate"] == 5.5
    assert result["n_groups"] == 2


def test_bh_adjustment_is_monotone_by_rank():
    p = np.array([.01, .04, .03, .2])
    q = benjamini_hochberg(p)
    assert np.all((0 <= q) & (q <= 1))
    assert q[0] <= q[2] <= q[1] <= q[3]
