"""Camelyon17-WILDS access with automatic downloads disabled."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .metadata import canonicalize_metadata


class HuggingFaceCamelyon17Adapter:
    """WILDS-compatible adapter over locally downloaded Hugging Face Parquet shards."""

    split_dict = {"train": 0, "id_val": 1, "test": 2, "val": 3}

    def __init__(self, root: str | Path):
        try:
            from datasets import concatenate_datasets, load_dataset
        except ImportError as error:
            raise ImportError("Install the local Parquet backend: pip install -e '.[hf]'") from error
        directory = Path(root)
        data_directory = directory / "data" if (directory / "data").is_dir() else directory
        files = {
            split: sorted(str(path) for path in data_directory.glob(f"{pattern}-*.parquet"))
            for split, pattern in {"train": "train", "validation": "validation", "test": "test"}.items()
        }
        missing = [split for split, paths in files.items() if not paths]
        if missing:
            raise FileNotFoundError(
                "Local Hugging Face Camelyon17-WILDS shards are incomplete. Missing "
                f"{missing} under {data_directory}. Download the repository from "
                "https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS to "
                f"{directory}. Automatic download is disabled."
            )
        parts = []
        for hf_split, paths in files.items():
            part = load_dataset("parquet", data_files={hf_split: paths}, split=hf_split)
            part = part.add_column("__hf_split", [hf_split] * len(part))
            parts.append(part)
        self.dataset = concatenate_datasets(parts)
        centers = np.asarray(self.dataset["center"], dtype=np.int64)
        hf_splits = np.asarray(self.dataset["__hf_split"], dtype=object)
        split_names = np.where(hf_splits == "train", "train", np.where(hf_splits == "test", "test", np.where(centers == 1, "val", "id_val")))
        self._split_names_array = split_names
        self.split_array = np.asarray([self.split_dict[str(name)] for name in split_names], dtype=np.int64)
        self.y_array = np.asarray(self.dataset["label"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[int(index)]
        metadata = np.asarray([int(row["center"]), int(row["slide"]), int(row["label"])], dtype=np.int64)
        return row["image"].convert("RGB"), int(row["label"]), metadata

    def metadata_frame(self) -> pd.DataFrame:
        columns = ["center", "slide", "patient", "node", "x_coord", "y_coord", "image_id"]
        available = [column for column in columns if column in self.dataset.column_names]
        frame = self.dataset.select_columns(available).to_pandas()
        frame.insert(0, "source_id", np.arange(len(frame), dtype=np.int64))
        frame["split"] = self._split_names_array
        frame["hospital_id"] = frame.pop("center")
        frame["slide_id"] = frame.pop("slide")
        frame["patient_id"] = frame.pop("patient").astype("string") if "patient" in frame else pd.NA
        frame["label"] = self.y_array
        return canonicalize_metadata(frame)


def require_dataset(root: str | Path) -> Path:
    expected = Path(root) / "camelyon17_v1.0"
    if not expected.exists():
        raise FileNotFoundError(
            "Camelyon17-WILDS is not present. Download it from "
            "https://wilds.stanford.edu/downloads/ and extract it to "
            f"{expected}. Automatic large-dataset download is disabled."
        )
    return expected


def load_wilds_dataset(config: Mapping[str, Any]):
    data = config.get("data", config)
    root = Path(data["root"])
    if "${MEDSTYLE_DATA_ROOT}" in str(root):
        raise EnvironmentError(
            "MEDSTYLE_DATA_ROOT is not set. On the server run: "
            "export MEDSTYLE_DATA_ROOT=/workspace/datasets"
        )
    backend = str(data.get("backend", "wilds")).lower()
    if backend == "huggingface_parquet":
        if data.get("download", False):
            raise ValueError("Large-dataset download is disabled; download Hugging Face shards manually")
        return HuggingFaceCamelyon17Adapter(root)
    if backend != "wilds":
        raise ValueError(f"Unknown Camelyon17 backend: {backend}")
    require_dataset(root)
    if data.get("download", False):
        raise ValueError("Large-dataset download is disabled; set data.download=false")
    try:
        from wilds import get_dataset
    except ImportError as error:
        raise ImportError("Install the optional training dependencies: pip install -e '.[train]'") from error
    return get_dataset(dataset="camelyon17", root_dir=str(root), download=False)


def extract_metadata(dataset: Any) -> pd.DataFrame:
    """Extract canonical metadata using WILDS' declared field names."""
    if hasattr(dataset, "metadata_frame"):
        return dataset.metadata_frame()
    metadata = np.asarray(dataset.metadata_array)
    fields = list(dataset.metadata_fields)
    if metadata.ndim != 2 or metadata.shape[1] != len(fields):
        raise ValueError("Unexpected WILDS metadata shape")
    frame = pd.DataFrame(metadata, columns=fields)
    private_metadata = getattr(dataset, "_metadata_df", None)
    if isinstance(private_metadata, pd.DataFrame) and len(private_metadata) == len(frame):
        for column in ["patient", "patient_id", "node", "x_coord", "y_coord", "image_id"]:
            if column in private_metadata and column not in frame:
                frame[column] = private_metadata[column].to_numpy()
    frame.insert(0, "source_id", np.arange(len(frame), dtype=np.int64))
    frame["label"] = np.asarray(dataset.y_array).reshape(-1)
    split_array = np.asarray(dataset.split_array).reshape(-1)
    split_names = {value: key for key, value in dataset.split_dict.items()}
    frame["split"] = [split_names.get(int(value), f"unknown:{value}") for value in split_array]
    return canonicalize_metadata(frame)


def subset(dataset: Any, split: str, transform: Any = None):
    return dataset.get_subset(split, transform=transform)
