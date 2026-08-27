import numpy as np
import pytest

from medstyleaudit.counterfactual.transplant import transplant
from medstyleaudit.counterfactual.feathering import cosine_alpha


@pytest.mark.parametrize("buffer,feather", [(0, 0), (0, 4), (8, 4), (16, 8)])
def test_transplant_is_pixel_exact_in_roi(buffer, feather):
    rng = np.random.default_rng(4)
    source = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    donor = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    result = transplant(source, donor, 32, buffer, feather)
    assert result.shape == source.shape
    assert np.array_equal(result[32:64, 32:64], source[32:64, 32:64])
    if buffer == 0 and feather == 0:
        assert np.array_equal(result[:32], donor[:32])


def test_transplant_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        transplant(np.zeros((96, 96, 3)), np.zeros((95, 96, 3)))


def test_feathering_uses_l_infinity_distance():
    alpha = cosine_alpha((96, 96), 32, 0, 4)
    # One step horizontally and one step diagonally are both d_infinity=1.
    assert alpha[31, 48] == alpha[31, 31]
