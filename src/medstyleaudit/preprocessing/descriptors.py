"""Mask-derived content descriptors used for primary matching."""

from __future__ import annotations

import numpy as np
from skimage.measure import label, perimeter, regionprops


DESCRIPTOR_COLUMNS = [
    "tissue_fraction", "component_count", "area_mean", "area_std", "area_max",
    "normalized_perimeter", "compactness", "eccentricity", "centroid_x", "centroid_y",
    "quadrant_0", "quadrant_1", "quadrant_2", "quadrant_3", "empty_mask",
    "grid_0", "grid_1", "grid_2", "grid_3", "grid_4", "grid_5", "grid_6", "grid_7", "grid_8",
]


def mask_descriptor(mask: np.ndarray) -> dict[str, float]:
    """Compute the prespecified geometric descriptor with finite empty sentinels."""
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    area = float(binary.sum())
    components = regionprops(label(binary))
    component_areas = np.asarray([region.area for region in components], dtype=float)
    total_perimeter = float(perimeter(binary))
    if area:
        centroid_y, centroid_x = np.argwhere(binary).mean(axis=0)
        eccentricity = float(sum(region.eccentricity * region.area for region in components) / area)
    else:
        centroid_y = centroid_x = -1.0
        eccentricity = -1.0
    quadrants = [
        binary[: height // 2, : width // 2], binary[: height // 2, width // 2 :],
        binary[height // 2 :, : width // 2], binary[height // 2 :, width // 2 :],
    ]
    grid = [
        binary[y0:y1, x0:x1]
        for y0, y1 in zip(np.linspace(0, height, 4, dtype=int)[:-1], np.linspace(0, height, 4, dtype=int)[1:])
        for x0, x1 in zip(np.linspace(0, width, 4, dtype=int)[:-1], np.linspace(0, width, 4, dtype=int)[1:])
    ]
    descriptor = {
        "tissue_fraction": area / binary.size,
        "component_count": float(len(components)),
        "area_mean": float(component_areas.mean()) / binary.size if component_areas.size else 0.0,
        "area_std": float(component_areas.std()) / binary.size if component_areas.size else 0.0,
        "area_max": float(component_areas.max()) / binary.size if component_areas.size else 0.0,
        "normalized_perimeter": total_perimeter / max(2 * (height + width), 1),
        "compactness": float(4 * np.pi * area / max(total_perimeter**2, 1.0)),
        "eccentricity": eccentricity,
        "centroid_x": float(centroid_x / width) if area else -1.0,
        "centroid_y": float(centroid_y / height) if area else -1.0,
        "empty_mask": float(not bool(area)),
    }
    descriptor.update({f"quadrant_{i}": float(part.mean()) for i, part in enumerate(quadrants)})
    descriptor.update({f"grid_{i}": float(part.mean()) for i, part in enumerate(grid)})
    if not all(np.isfinite(value) for value in descriptor.values()):
        raise FloatingPointError("Descriptor contains non-finite values")
    return descriptor


def descriptor_vector(descriptor: dict[str, float]) -> np.ndarray:
    return np.asarray([descriptor[column] for column in DESCRIPTOR_COLUMNS], dtype=np.float64)
