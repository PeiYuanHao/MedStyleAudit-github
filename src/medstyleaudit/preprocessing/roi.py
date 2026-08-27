"""Exact central ROI geometry and masking."""

from __future__ import annotations

import numpy as np


def roi_slices(shape: tuple[int, ...], roi_size: int = 32) -> tuple[slice, slice]:
    """Return centered spatial slices for HWC or CHW arrays."""
    height, width = shape[-2:] if len(shape) >= 3 and shape[-1] != 3 else shape[:2]
    if roi_size <= 0 or roi_size > min(height, width):
        raise ValueError(f"Invalid ROI size {roi_size} for spatial shape {(height, width)}")
    top, left = (height - roi_size) // 2, (width - roi_size) // 2
    return slice(top, top + roi_size), slice(left, left + roi_size)


def protected_mask(shape: tuple[int, int], roi_size: int = 32, buffer: int = 0) -> np.ndarray:
    height, width = shape
    if buffer < 0:
        raise ValueError("buffer must be non-negative")
    y, x = roi_slices((height, width), roi_size)
    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, y.start - buffer) : min(height, y.stop + buffer), max(0, x.start - buffer) : min(width, x.stop + buffer)] = True
    return mask


def roi_only(image: np.ndarray, roi_size: int = 32, fill_value: float | int = 0) -> np.ndarray:
    result = np.full_like(image, fill_value)
    y, x = roi_slices(image.shape, roi_size)
    if image.ndim == 3 and image.shape[-1] in {1, 3, 4}:
        result[y, x, :] = image[y, x, :]
    else:
        result[..., y, x] = image[..., y, x]
    return result
