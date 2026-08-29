"""Frozen-model inference on original/within/cross counterfactual triplets."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from medstyleaudit.counterfactual.qa import roi_identity_metrics, seam_metrics, tensor_roi_identity_metrics
from medstyleaudit.counterfactual.transplant import transplant
from medstyleaudit.preprocessing.roi import roi_only


class ROIIdentityError(AssertionError):
    """Raised with the QA ledger when image or final-tensor ROI identity fails."""

    def __init__(self, message: str, qa: pd.DataFrame):
        super().__init__(message)
        self.qa = qa


SOURCE_IDENTITY_COLUMNS = (
    "source_split",
    "source_hospital",
    "source_slide",
    "source_patient",
    "source_physical_id",
    "label",
)


def _metadata_equal(left: object, right: object) -> bool:
    try:
        if bool(pd.isna(left)) and bool(pd.isna(right)):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


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
    batch_size: int = 64,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a frozen model and return complete prediction and construction-QA ledgers."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.to(device).eval()
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    prediction_rows, qa_rows = [], []
    pending_tensors, pending_bases = [], []
    canonical_original_logits: dict[object, float] = {}
    source_identities: dict[object, dict[str, object]] = {}

    def flush() -> None:
        if not pending_tensors:
            return
        tensors = torch.stack(pending_tensors).reshape(-1, *pending_tensors[0].shape[1:])
        tensors = tensors.to(device, non_blocking=cuda)
        logits = model(tensors).reshape(len(pending_tensors), 3).detach().cpu().numpy()
        for base, values in zip(pending_bases, logits):
            source_id = base["source_id"]
            if source_id not in canonical_original_logits:
                canonical_original_logits[source_id] = float(values[0])
            prediction_rows.append({
                **base,
                "original_logit": canonical_original_logits[source_id],
                "within_logit": float(values[1]),
                "cross_logit": float(values[2]),
                "source_buffer": source_buffer,
                "feather_width": feather_width,
                "roi_only_control": roi_only_control,
            })
        pending_tensors.clear()
        pending_bases.clear()

    with torch.no_grad():
        records = triplets.itertuples(index=False)
        if show_progress:
            from tqdm.auto import tqdm
            records = tqdm(records, total=len(triplets), desc="Counterfactual audit inference", unit="triplet")
        for record in records:
            base = {name: getattr(record, name) for name in triplets.columns}
            source_id = base["source_id"]
            identity = {column: base[column] for column in SOURCE_IDENTITY_COLUMNS if column in base}
            if source_id in source_identities:
                for column, value in identity.items():
                    if not _metadata_equal(value, source_identities[source_id][column]):
                        raise ValueError(
                            f"Inconsistent source metadata for source_id {source_id}: {column}"
                        )
            else:
                source_identities[source_id] = identity
            source = np.asarray(dataset[int(record.source_id)][0].convert("RGB"))
            within_donor = np.asarray(dataset[int(record.within_donor)][0].convert("RGB"))
            cross_donor = np.asarray(dataset[int(record.cross_donor)][0].convert("RGB"))
            within = transplant(source, within_donor, roi_size, source_buffer, feather_width)
            cross = transplant(source, cross_donor, roi_size, source_buffer, feather_width)
            if roi_only_control:
                source, within, cross = (roi_only(image, roi_size, 0) for image in (source, within, cross))
            image_rows = [
                {"triplet_id": record.triplet_id, "arm": arm, **roi_identity_metrics(source, composite, roi_size), **seam_metrics(composite, roi_size, source_buffer)}
                for arm, composite in (("within", within), ("cross", cross))
            ]
            if any(not row["roi_equal"] for row in image_rows):
                qa_rows.extend(image_rows)
                raise ROIIdentityError(f"Image-level ROI identity failed for triplet {record.triplet_id}", pd.DataFrame(qa_rows))
            tensors = torch.stack([transform(image) for image in (source, within, cross)])
            tensor_rows = [
                tensor_roi_identity_metrics(tensors[0], tensors[index], roi_size)
                for index in (1, 2)
            ]
            for image_row, tensor_row in zip(image_rows, tensor_rows):
                image_row.update(tensor_row)
            qa_rows.extend(image_rows)
            if any(not row["tensor_roi_equal"] for row in tensor_rows):
                raise ROIIdentityError(f"Final classifier-input tensor ROI identity failed for triplet {record.triplet_id}", pd.DataFrame(qa_rows))
            pending_tensors.append(tensors)
            pending_bases.append(base)
            if len(pending_tensors) >= batch_size:
                flush()
        flush()
    qa = pd.DataFrame(qa_rows)
    return pd.DataFrame(prediction_rows), qa
