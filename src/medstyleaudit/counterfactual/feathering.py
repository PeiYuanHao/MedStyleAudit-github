"""Cosine alpha masks for source-protected compositing."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_cdt

from medstyleaudit.preprocessing.roi import protected_mask


def cosine_alpha(shape: tuple[int, int], roi_size: int = 32, buffer: int = 0, feather_width: int = 0) -> np.ndarray:
    """Return alpha=1 on the protected set and cosine decay outside it."""
    protected = protected_mask(shape, roi_size, buffer)
    alpha = protected.astype(np.float64)
    if feather_width <= 0:
        return alpha
    # Equation (5) in the manuscript uses Chebyshev/L-infinity distance.
    distance = distance_transform_cdt(~protected, metric="chessboard").astype(float)
    ring = (distance > 0) & (distance < feather_width)
    alpha[ring] = 0.5 * (1.0 + np.cos(np.pi * distance[ring] / feather_width))
    return alpha
