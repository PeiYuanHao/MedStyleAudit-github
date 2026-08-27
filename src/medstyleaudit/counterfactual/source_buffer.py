"""Protected ROI plus source-buffer masks."""

from medstyleaudit.preprocessing.roi import protected_mask


def intervention_area_ratio(shape: tuple[int, int] = (96, 96), roi_size: int = 32, buffer: int = 0) -> float:
    """Return A(r)=|Omega minus P_r| / |Omega minus ROI| from the manuscript."""
    roi = protected_mask(shape, roi_size, 0)
    protected = protected_mask(shape, roi_size, buffer)
    return float((~protected).sum() / max((~roi).sum(), 1))

__all__ = ["protected_mask", "intervention_area_ratio"]
