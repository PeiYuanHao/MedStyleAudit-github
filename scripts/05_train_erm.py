"""Phase 5: random-initialized ERM training with persistent checkpoints."""

from pathlib import Path

import numpy as np

from medstyleaudit.data.wilds_loader import load_wilds_dataset
from medstyleaudit.data.loading import loader_kwargs
from medstyleaudit.models.trainer import build_model, evaluate, train
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.seed import seed_everything
from medstyleaudit.utils.paths import configured_output, experiment_path


def main() -> None:
    parser = common_parser("Train an ERM tumor classifier", "configs/models/resnet50.yaml")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--predictions-only", action="store_true")
    args = parser.parse_args(); config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42)); seed_everything(seed)
    architecture = config["model"]["architecture"]
    output_root = args.output_dir or configured_output(config, f"checkpoints/{architecture}")
    output = output_root / f"seed_{seed:04d}"
    if args.resume: config["training"]["resume"] = str(args.resume)
    dataset = load_wilds_dataset(config)
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import Compose, Normalize, ToTensor
    transform = Compose([ToTensor(), Normalize(config["model"]["input_mean"], config["model"]["input_std"])])

    class IndexedSplit(Dataset):
        def __init__(self, indices): self.indices = np.asarray(indices)
        def __len__(self): return len(self.indices)
        def __getitem__(self, index):
            source_id = int(self.indices[index]); image, target, _ = dataset[source_id]
            return transform(image.convert("RGB")), torch.as_tensor(target), source_id

    train_value, val_value, id_val_value, test_value = dataset.split_dict["train"], dataset.split_dict["val"], dataset.split_dict["id_val"], dataset.split_dict["test"]
    train_indices = np.flatnonzero(np.asarray(dataset.split_array) == train_value)
    val_indices = np.flatnonzero(np.asarray(dataset.split_array) == val_value)
    id_val_indices = np.flatnonzero(np.asarray(dataset.split_array) == id_val_value)
    test_indices = np.flatnonzero(np.asarray(dataset.split_array) == test_value)
    if args.dry_run:
        train_indices, val_indices, id_val_indices = train_indices[:256], val_indices[:128], id_val_indices[:128]
    settings = config["training"]
    data_loader_kwargs = loader_kwargs(settings, args.device)
    train_loader = DataLoader(IndexedSplit(train_indices), shuffle=True, **data_loader_kwargs)
    val_loader = DataLoader(IndexedSplit(val_indices), shuffle=False, **data_loader_kwargs)
    model = build_model(config)
    if args.predictions_only:
        if not args.allow_final_test:
            raise PermissionError("--predictions-only requires --allow-final-test")
        from medstyleaudit.utils.cli import enforce_final_test_guard
        from medstyleaudit.protocol import assert_frozen_execution_config
        enforce_final_test_guard("test", True)
        assert_frozen_execution_config(model=config, seed=seed)
        history = []
    else:
        run = start_run(f"train_erm_{architecture}", config, output, seed, overwrite=args.overwrite)
        history = train(model, train_loader, val_loader, config, output, args.device, args.dry_run)
    best_state = torch.load(output / "best.ckpt", map_location=args.device); model.load_state_dict(best_state["model"])
    inference_options = {
        "amp": bool(settings.get("amp", False)),
        "channels_last": bool(settings.get("channels_last", False)),
    }
    _, id_predictions = evaluate(model, DataLoader(IndexedSplit(id_val_indices), shuffle=False, **data_loader_kwargs), args.device, True, "ID validation predictions", **inference_options)
    from medstyleaudit.utils.io import save_table
    predictions_dir = experiment_path(f"predictions/{architecture}/seed_{seed:04d}")
    id_predictions["split"] = "id_val"; save_table(id_predictions, predictions_dir / "id_val.parquet")
    _, ood_predictions = evaluate(model, val_loader, args.device, True, "OOD validation predictions", **inference_options)
    save_table(ood_predictions.assign(split="ood_val"), predictions_dir / "ood_val.parquet")
    if args.predictions_only:
        from medstyleaudit.protocol import record_final_test_opened
        test_loader = DataLoader(IndexedSplit(test_indices), shuffle=False, **data_loader_kwargs)
        _, final_predictions = evaluate(model, test_loader, args.device, True, "Final OOD test predictions", **inference_options)
        save_table(final_predictions.assign(split="ood_test"), predictions_dir / "ood_test.parquet")
        marker = record_final_test_opened(experiment_path(""), f"predict-{architecture}-{seed:04d}")
        print(f"Final-test predictions written; opening record: {marker}")
    else:
        run.complete(status="completed", epochs_completed=len(history), best_checkpoint=str((output / "best.ckpt").resolve()))


if __name__ == "__main__": main()
