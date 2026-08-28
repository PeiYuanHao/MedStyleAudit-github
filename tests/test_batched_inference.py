import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from medstyleaudit.audit.inference import infer_triplets


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


def test_triplet_inference_batches_model_calls_without_changing_row_count():
    transform = lambda image: torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    triplets = pd.DataFrame([
        {"triplet_id": f"t{index}", "source_id": 0, "within_donor": 1, "cross_donor": 2 + index % 2, "source_split": "val"}
        for index in range(5)
    ])
    model = _RecordingModel()
    predictions, qa = infer_triplets(model, _Dataset(), triplets, transform, batch_size=2)
    assert model.batch_sizes == [6, 6, 3]
    assert len(predictions) == 5
    assert len(qa) == 10
