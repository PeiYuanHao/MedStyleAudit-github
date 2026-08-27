import numpy as np

from medstyleaudit.preprocessing.roi import protected_mask, roi_only, roi_slices


def test_central_roi_coordinates():
    y, x = roi_slices((96, 96, 3), 32)
    assert (y.start, y.stop, x.start, x.stop) == (32, 64, 32, 64)


def test_roi_only_preserves_exact_pixels():
    image = np.arange(96 * 96 * 3, dtype=np.int32).reshape(96, 96, 3)
    result = roi_only(image, 32, -1)
    assert np.array_equal(result[32:64, 32:64], image[32:64, 32:64])
    assert (result[~protected_mask((96, 96), 32)] == -1).all()
