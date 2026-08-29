import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from medstyleaudit.audit.inference import infer_triplets
from medstyleaudit.preprocessing.roi import roi_only


class _Dataset:
    def __init__(self):
        rng = np.random.default_rng(9)
        self.images = [Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)) for _ in range(4)]

    def __getitem__(self, index):
        return self.images[index], 0, None


class _RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, values):
        self.batch_sizes.append(len(values))
        return values.mean(dim=(1, 2, 3))


class _DriftingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = 0

    def forward(self, values):
        offsets = torch.arange(
            self.seen, self.seen + len(values), device=values.device, dtype=values.dtype
        )
        self.seen += len(values)
        return values.mean(dim=(1, 2, 3)) + offsets / 1000


def _transform(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255


def test_triplet_inference_batches_model_calls_without_changing_row_count():
    triplets = pd.DataFrame([
        {"triplet_id": f"t{index}", "source_id": 0, "within_donor": 1, "cross_donor": 2 + index % 2, "source_split": "val"}
        for index in range(5)
    ])
    model = _RecordingModel()
    predictions, qa = infer_triplets(model, _Dataset(), triplets, _transform, batch_size=2)
    assert model.batch_sizes == [6, 6, 3]
    assert len(predictions) == 5
    assert len(qa) == 10


def test_repeated_source_uses_exact_canonical_logit_across_batches():
    triplets = pd.DataFrame([
        {
            "triplet_id": f"t{index}",
            "source_id": 0,
            "within_donor": 1 + index % 2,
            "cross_donor": 2 + index % 2,
            "source_split": "val",
            "source_hospital": 1,
            "target_hospital": 4,
        }
        for index in range(5)
    ])
    predictions, _ = infer_triplets(
        _DriftingModel(), _Dataset(), triplets, _transform, batch_size=2
    )
    assert predictions["original_logit"].nunique() == 1
    assert predictions["within_logit"].nunique() == len(triplets)
    assert predictions["cross_logit"].nunique() == len(triplets)


def test_roi_only_builds_its_own_canonical_source_anchor():
    dataset = _Dataset()
    triplets = pd.DataFrame([
        {
            "triplet_id": f"t{index}",
            "source_id": 0,
            "within_donor": 1,
            "cross_donor": 2,
            "source_split": "val",
        }
        for index in range(3)
    ])
    full, _ = infer_triplets(_RecordingModel(), dataset, triplets, _transform, batch_size=2)
    roi, _ = infer_triplets(
        _RecordingModel(), dataset, triplets, _transform, batch_size=2, roi_only_control=True
    )
    expected = float(_transform(roi_only(np.asarray(dataset[0][0]), 32, 0)).mean())
    assert roi["original_logit"].tolist() == [expected] * len(triplets)
    assert roi.loc[0, "original_logit"] != full.loc[0, "original_logit"]


def test_repeated_source_with_inconsistent_metadata_fails_closed():
    triplets = pd.DataFrame([
        {
            "triplet_id": "t0",
            "source_id": 0,
            "within_donor": 1,
            "cross_donor": 2,
            "source_split": "val",
            "source_hospital": 1,
        },
        {
            "triplet_id": "t1",
            "source_id": 0,
            "within_donor": 2,
            "cross_donor": 3,
            "source_split": "test",
            "source_hospital": 1,
        },
    ])
    with pytest.raises(ValueError, match="Inconsistent source metadata.*source_split"):
        infer_triplets(_RecordingModel(), _Dataset(), triplets, _transform)
