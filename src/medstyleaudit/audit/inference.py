"""Frozen-model inference on original/within/cross counterfactual triplets."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from medstyleaudit.counterfactual.qa import roi_identity_metrics, seam_metrics
from medstyleaudit.counterfactual.transplant import transplant
from medstyleaudit.preprocessing.roi import roi_only


def infer_triplets(
    model,
    dataset,
    triplets: pd.DataFrame,
    transform: Callable,
    *,
    device: str = "cpu",
    roi_size: int = 32,
    source_buffer: int = 0,
    feather_width: int = 4,
    roi_only_control: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a frozen model and return complete prediction and construction-QA ledgers."""
    import torch

    model.to(device).eval()
    prediction_rows, qa_rows = [], []
    with torch.no_grad():
        for record in triplets.itertuples(index=False):
            source = np.asarray(dataset[int(record.source_id)][0].convert("RGB"))
            within_donor = np.asarray(dataset[int(record.within_donor)][0].convert("RGB"))
            cross_donor = np.asarray(dataset[int(record.cross_donor)][0].convert("RGB"))
            within = transplant(source, within_donor, roi_size, source_buffer, feather_width)
            cross = transplant(source, cross_donor, roi_size, source_buffer, feather_width)
            if roi_only_control:
                source, within, cross = (roi_only(image, roi_size, 0) for image in (source, within, cross))
            tensors = torch.stack([transform(image) for image in (source, within, cross)]).to(device)
            logits = model(tensors).reshape(-1).detach().cpu().numpy()
            base = {name: getattr(record, name) for name in triplets.columns}
            prediction_rows.append({**base, "original_logit": float(logits[0]), "within_logit": float(logits[1]), "cross_logit": float(logits[2]), "source_buffer": source_buffer, "feather_width": feather_width, "roi_only_control": roi_only_control})
            qa_rows.append({"triplet_id": record.triplet_id, "arm": "within", **roi_identity_metrics(source, within, roi_size), **seam_metrics(within, roi_size, source_buffer)})
            qa_rows.append({"triplet_id": record.triplet_id, "arm": "cross", **roi_identity_metrics(source, cross, roi_size), **seam_metrics(cross, roi_size, source_buffer)})
    qa = pd.DataFrame(qa_rows)
    if not qa.empty and ((qa["max_roi_difference"] != 0).any() or (qa["mean_roi_difference"] != 0).any()):
        raise AssertionError("Nonzero tensor-level ROI difference detected")
    return pd.DataFrame(prediction_rows), qa
