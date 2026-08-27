"""Synthetic peripheral cue insertion for positive calibration."""

from __future__ import annotations

import numpy as np

from medstyleaudit.preprocessing.roi import roi_slices


def plant_corner_cue(image: np.ndarray, label: int, strength: float, cue_size: int = 8, rng: np.random.Generator | None = None) -> np.ndarray:
    """Insert a label-correlated RGB cue outside the ROI with probability strength."""
    if not 0 <= strength <= 1:
        raise ValueError("strength must lie in [0, 1]")
    rng = rng or np.random.default_rng()
    result = np.array(image, copy=True)
    if rng.random() <= strength:
        value = 255 if int(label) else 0
        if result.ndim == 3 and result.shape[-1] in {1, 3, 4}:
            result[:cue_size, :cue_size, ...] = value
        else:
            result[..., :cue_size, :cue_size] = value
    y, x = roi_slices(image.shape, 32)
    original_roi = image[y, x, ...] if image.ndim == 3 and image.shape[-1] in {1, 3, 4} else image[..., y, x]
    new_roi = result[y, x, ...] if image.ndim == 3 and image.shape[-1] in {1, 3, 4} else result[..., y, x]
    if not np.array_equal(original_roi, new_roi):
        raise AssertionError("Planted shortcut modified the ROI")
    return result
