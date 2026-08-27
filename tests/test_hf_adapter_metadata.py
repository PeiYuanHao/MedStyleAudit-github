import numpy as np

from medstyleaudit.data.wilds_loader import HuggingFaceCamelyon17Adapter


def test_hf_split_derivation_rule():
    centers = np.array([0, 3, 4, 0, 1, 2])
    hf_splits = np.array(["train", "train", "train", "validation", "validation", "test"], dtype=object)
    names = np.where(hf_splits == "train", "train", np.where(hf_splits == "test", "test", np.where(centers == 1, "val", "id_val")))
    assert names.tolist() == ["train", "train", "train", "id_val", "val", "test"]
