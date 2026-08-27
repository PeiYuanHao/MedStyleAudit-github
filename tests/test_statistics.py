import numpy as np
import pandas as pd

from medstyleaudit.statistics.bootstrap import crossed_multiplier_bootstrap
from medstyleaudit.statistics.multiple_testing import benjamini_hochberg


def test_crossed_bootstrap_is_reproducible():
    frame = pd.DataFrame({"value": [1., 2., 3.], "source": ["a", "b", "c"], "donor": ["x", "x", "y"]})
    first = crossed_multiplier_bootstrap(frame, "value", ["source", "donor"], draws=100, seed=9)
    second = crossed_multiplier_bootstrap(frame, "value", ["source", "donor"], draws=100, seed=9)
    assert first == second


def test_bh_adjustment_is_monotone_by_rank():
    p = np.array([.01, .04, .03, .2])
    q = benjamini_hochberg(p)
    assert np.all((0 <= q) & (q <= 1))
    assert q[0] <= q[2] <= q[1] <= q[3]
