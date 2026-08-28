"""Fixed HSV tissue mask and stability metrics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.ndimage import median_filter
from skimage.color import hed2rgb, rgb2hed, rgb2hsv
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects


def tissue_mask(image: np.ndarray, config: Mapping[str, Any] | None = None) -> np.ndarray:
    """Create a hospital-agnostic tissue mask from an RGB image."""
    settings = dict(config or {})
    saturation_min = float(settings.get("saturation_min", 0.05))
    value_max = float(settings.get("value_max", 0.95))
    min_object_size = int(settings.get("min_object_size", 8))
    closing_radius = int(settings.get("closing_radius", 1))
    median_size = int(settings.get("median_size", 3))
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.max(initial=0) > 1:
        rgb /= 255.0
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("tissue_mask expects an HxWx3 RGB image")
    if median_size > 1:
        rgb = median_filter(rgb, size=(median_size, median_size, 1), mode="nearest")
    hsv = rgb2hsv(np.clip(rgb, 0, 1))
    mask = (hsv[..., 1] >= saturation_min) & (hsv[..., 2] <= value_max)
    if closing_radius:
        mask = binary_closing(mask, disk(closing_radius))
    mask = remove_small_objects(mask, min_size=min_object_size)
    return remove_small_holes(mask, area_threshold=min_object_size)


def overlap_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return {
        "dice": float(2 * intersection / max(a.sum() + b.sum(), 1)),
        "jaccard": float(intersection / max(union, 1)),
    }


def mask_stability(image: np.ndarray, base: Mapping[str, Any], perturbations: list[Mapping[str, Any]]) -> list[dict[str, float]]:
    reference = tissue_mask(image, base)
    rows = []
    for index, update in enumerate(perturbations):
        settings = {**base, **update}
        rows.append({"perturbation": index, **overlap_metrics(reference, tissue_mask(image, settings))})
    return rows


def hed_appearance_perturbation(image: np.ndarray, h_scale: float = 1.0, e_scale: float = 1.0) -> np.ndarray:
    """Apply a deterministic H/E channel scaling for mask-stability diagnostics."""
    rgb = np.asarray(image, dtype=np.float64)
    integer_input = np.issubdtype(np.asarray(image).dtype, np.integer)
    if rgb.max(initial=0) > 1:
        rgb /= 255.0
    hed = rgb2hed(np.clip(rgb, 0, 1))
    hed[..., 0] *= float(h_scale); hed[..., 1] *= float(e_scale)
    perturbed = np.clip(hed2rgb(hed), 0, 1)
    return np.rint(perturbed * 255).astype(np.uint8) if integer_input else perturbed.astype(np.asarray(image).dtype)
