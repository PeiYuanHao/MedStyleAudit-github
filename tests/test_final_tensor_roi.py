import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from medstyleaudit.audit.inference import ROIIdentityError, infer_triplets


class _Dataset:
    def __init__(self):
        rng = np.random.default_rng(3)
        self.images = [Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)) for _ in range(3)]

    def __getitem__(self, index):
        return self.images[index], 0, None


class _Model(torch.nn.Module):
    def forward(self, values):
        return values.mean(dim=(1, 2, 3))


def test_final_tensor_roi_mutation_is_rejected():
    calls = {"n": 0}

    def transform(image):
        calls["n"] += 1
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        if calls["n"] == 2:
            tensor[0, 40, 40] += 1
        return tensor

    triplets = pd.DataFrame([{"triplet_id": "t1", "source_id": 0, "within_donor": 1, "cross_donor": 2, "source_split": "val", "source_hospital": 1, "target_hospital": 0, "label": 1}])
    with pytest.raises(ROIIdentityError) as caught:
        infer_triplets(_Model(), _Dataset(), triplets, transform)
    assert (caught.value.qa["tensor_roi_equal"] == 0).any()
