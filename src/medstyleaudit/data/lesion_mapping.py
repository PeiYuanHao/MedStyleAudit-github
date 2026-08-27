"""CAMELYON XML lesion annotation parsing and patch projection."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import numpy as np
from skimage.draw import polygon
from skimage.measure import label, regionprops


def read_camelyon_xml(path: str | Path) -> list[np.ndarray]:
    """Read annotation coordinate polygons without assuming WSI orientation."""
    root = ET.parse(path).getroot()
    polygons = []
    for annotation in root.findall(".//Annotation"):
        points = []
        for coordinate in annotation.findall(".//Coordinate"):
            points.append((float(coordinate.attrib["X"]), float(coordinate.attrib["Y"])))
        if len(points) >= 3:
            polygons.append(np.asarray(points, dtype=np.float64))
    return polygons


def project_polygons(
    polygons: Sequence[np.ndarray],
    patch_origin_xy: tuple[float, float],
    patch_size: int = 96,
    scale: float = 1.0,
    orientation: str = "identity",
    coordinate_reference: str = "top_left",
) -> np.ndarray:
    """Rasterize WSI-level polygons into patch coordinates at an explicit scale."""
    if orientation != "identity":
        raise ValueError("Only explicitly validated identity WSI orientation is supported")
    if coordinate_reference != "top_left":
        raise ValueError("Only explicitly configured top-left patch origins are supported")
    if scale <= 0:
        raise ValueError("scale must be positive")
    mask = np.zeros((patch_size, patch_size), dtype=bool)
    origin_x, origin_y = patch_origin_xy
    for coordinates in polygons:
        local_x = (coordinates[:, 0] - origin_x) / scale
        local_y = (coordinates[:, 1] - origin_y) / scale
        rr, cc = polygon(local_y, local_x, shape=mask.shape)
        mask[rr, cc] = True
    return mask


def lesion_features(mask: np.ndarray, roi_size: int = 32, source_buffer: int = 0) -> dict[str, float | int]:
    """Compute peripheral lesion features after excluding the protected square."""
    height, width = mask.shape
    y0, x0 = (height - roi_size) // 2, (width - roi_size) // 2
    protected = np.zeros_like(mask, dtype=bool)
    y1, y2 = max(0, y0 - source_buffer), min(height, y0 + roi_size + source_buffer)
    x1, x2 = max(0, x0 - source_buffer), min(width, x0 + roi_size + source_buffer)
    protected[y1:y2, x1:x2] = True
    peripheral = np.asarray(mask, dtype=bool) & ~protected
    components = regionprops(label(peripheral))
    coords = np.argwhere(peripheral)
    if coords.size:
        py = np.clip(coords[:, 0], y1, y2 - 1)
        px = np.clip(coords[:, 1], x1, x2 - 1)
        distance = np.sqrt((coords[:, 0] - py) ** 2 + (coords[:, 1] - px) ** 2).min()
    else:
        distance = float("inf")
    return {
        "peripheral_tumor_presence": int(peripheral.any()),
        "peripheral_tumor_fraction": float(peripheral.mean()),
        "peripheral_tumor_distance": float(distance),
        "peripheral_largest_component": float(max((region.area for region in components), default=0)),
        "transplanted_area_tumor_fraction": float(peripheral.sum() / max((~protected).sum(), 1)),
    }


def validate_alignment(records: Sequence[tuple[np.ndarray, int]], roi_size: int = 32, minimum_agreement: float = 0.99) -> dict[str, float | str | int]:
    """Compare mapped central tumor presence with released patch labels."""
    if not records:
        return {"status": "unavailable", "reason": "no mapped annotations", "n": 0}
    matches = []
    for mask, target in records:
        y0 = (mask.shape[0] - roi_size) // 2
        x0 = (mask.shape[1] - roi_size) // 2
        mapped = int(mask[y0 : y0 + roi_size, x0 : x0 + roi_size].any())
        matches.append(mapped == int(target))
    agreement = float(np.mean(matches))
    return {"status": "validated" if agreement >= minimum_agreement else "failed", "agreement": agreement, "n": len(matches), "threshold": minimum_agreement}
