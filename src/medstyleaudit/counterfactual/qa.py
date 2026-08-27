"""Tensor identity and seam diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.filters import sobel

from medstyleaudit.preprocessing.roi import protected_mask, roi_slices


def roi_identity_metrics(source: np.ndarray, composite: np.ndarray, roi_size: int = 32) -> dict[str, float]:
    y, x = roi_slices(source.shape, roi_size)
    channel_last = source.ndim == 3 and source.shape[-1] in {1, 3, 4}
    first = source[y, x, :] if channel_last else source[..., y, x]
    second = composite[y, x, :] if channel_last else composite[..., y, x]
    difference = np.abs(first.astype(float) - second.astype(float))
    equal = np.array_equal(first, second)
    return {"max_roi_difference": float(difference.max(initial=0)), "mean_roi_difference": float(difference.mean()), "roi_equal": float(equal), "roi_ssim": 1.0 if equal else float("nan")}


def seam_metrics(image: np.ndarray, roi_size: int = 32, buffer: int = 0, ring_width: int = 1) -> dict[str, float]:
    channel_last = image.ndim == 3 and image.shape[-1] in {1, 3, 4}
    spatial = image.shape[:2] if channel_last else image.shape[-2:]
    protected = protected_mask(spatial, roi_size, buffer)
    outer = binary_dilation(protected, iterations=ring_width) & ~protected
    inner = protected & ~binary_erosion(protected, iterations=ring_width)
    gray = image.astype(float).mean(axis=-1 if channel_last else 0)
    gradient = sobel(gray)
    return {
        "seam_gradient": float(gradient[outer].mean()) if outer.any() else 0.0,
        "cross_boundary_intensity_jump": float(abs(gray[inner].mean() - gray[outer].mean())) if inner.any() and outer.any() else 0.0,
        "edge_magnitude": float(gradient[np.logical_or(inner, outer)].mean()),
    }
