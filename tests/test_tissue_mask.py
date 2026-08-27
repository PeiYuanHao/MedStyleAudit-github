import numpy as np

from medstyleaudit.preprocessing.tissue_mask import hed_appearance_perturbation, overlap_metrics, tissue_mask


def test_fixed_mask_distinguishes_colored_tissue_from_white():
    image = np.full((96, 96, 3), 255, np.uint8)
    image[20:80, 20:80] = [160, 40, 100]
    mask = tissue_mask(image)
    assert mask[48, 48]
    assert not mask[0, 0]


def test_overlap_identity():
    mask = np.eye(8, dtype=bool)
    assert overlap_metrics(mask, mask) == {"dice": 1.0, "jaccard": 1.0}


def test_hed_perturbation_preserves_shape_and_type():
    image = np.full((16, 16, 3), [170, 80, 120], dtype=np.uint8)
    perturbed = hed_appearance_perturbation(image, 1.1, .9)
    assert perturbed.shape == image.shape and perturbed.dtype == image.dtype
