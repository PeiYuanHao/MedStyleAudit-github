import numpy as np

from medstyleaudit.preprocessing.descriptors import DESCRIPTOR_COLUMNS, mask_descriptor


def test_empty_descriptor_is_finite():
    descriptor = mask_descriptor(np.zeros((96, 96), dtype=bool))
    assert set(DESCRIPTOR_COLUMNS) == set(descriptor)
    assert all(np.isfinite(value) for value in descriptor.values())
    assert descriptor["empty_mask"] == 1


def test_simple_square_descriptor():
    mask = np.zeros((96, 96), bool); mask[24:72, 24:72] = True
    descriptor = mask_descriptor(mask)
    assert descriptor["tissue_fraction"] == 0.25
    assert descriptor["component_count"] == 1
    assert descriptor["empty_mask"] == 0
