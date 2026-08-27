"""Peripheral context transplantation with exact final-tensor ROI restoration."""

from __future__ import annotations

import numpy as np

from medstyleaudit.preprocessing.roi import roi_slices
from .feathering import cosine_alpha


def transplant(source: np.ndarray, donor: np.ndarray, roi_size: int = 32, buffer: int = 0, feather_width: int = 0) -> np.ndarray:
    """Compose equal-shaped HWC or CHW arrays and restore protected source pixels."""
    source_array, donor_array = np.asarray(source), np.asarray(donor)
    if source_array.shape != donor_array.shape:
        raise ValueError(f"Source and donor shapes differ: {source_array.shape} vs {donor_array.shape}")
    channel_last = source_array.ndim == 3 and source_array.shape[-1] in {1, 3, 4}
    spatial = source_array.shape[:2] if channel_last else source_array.shape[-2:]
    alpha = cosine_alpha(spatial, roi_size, buffer, feather_width)
    broadcast = alpha[..., None] if channel_last else alpha[None, ...]
    result = broadcast * source_array.astype(np.float64) + (1 - broadcast) * donor_array.astype(np.float64)
    if np.issubdtype(source_array.dtype, np.integer):
        result = np.rint(result).clip(np.iinfo(source_array.dtype).min, np.iinfo(source_array.dtype).max).astype(source_array.dtype)
    else:
        result = result.astype(source_array.dtype, copy=False)
    y, x = roi_slices(source_array.shape, roi_size)
    if channel_last:
        result[y, x, :] = source_array[y, x, :]
    else:
        result[..., y, x] = source_array[..., y, x]
    assert_roi_identity(source_array, result, roi_size)
    return result


def assert_roi_identity(source: np.ndarray, composite: np.ndarray, roi_size: int = 32) -> None:
    y, x = roi_slices(source.shape, roi_size)
    if source.ndim == 3 and source.shape[-1] in {1, 3, 4}:
        first, second = source[y, x, :], composite[y, x, :]
    else:
        first, second = source[..., y, x], composite[..., y, x]
    if not np.array_equal(first, second):
        difference = np.abs(first.astype(float) - second.astype(float))
        raise AssertionError(f"ROI identity failure: max={difference.max()}, mean={difference.mean()}")
