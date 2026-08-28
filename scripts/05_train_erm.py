"""Phase 5: random-initialized ERM training with persistent checkpoints."""

from pathlib import Path

import numpy as np

from medstyleaudit.data.wilds_loader import load_wilds_dataset
from medstyleaudit.models.trainer import build_model, evaluate, train
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.seed import seed_everything
from medstyleaudit.utils.paths import configured_output


def main() -> None:
    parser = common_parser("Train an ERM tumor classifier", "configs/models/resnet50.yaml")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args(); config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42)); seed_everything(seed)
    architecture = config["model"]["architecture"]
    output_root = args.output_dir or configured_output(config, f"checkpoints/{architecture}")
    output = output_root / f"seed_{seed:04d}"
    if args.resume: config["training"]["resume"] = str(args.resume)
    run = start_run(f"train_erm_{architecture}", config, output, seed, overwrite=args.overwrite)
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

    train_value, val_value, id_val_value = dataset.split_dict["train"], dataset.split_dict["val"], dataset.split_dict["id_val"]
    train_indices = np.flatnonzero(np.asarray(dataset.split_array) == train_value)
    val_indices = np.flatnonzero(np.asarray(dataset.split_array) == val_value)
    id_val_indices = np.flatnonzero(np.asarray(dataset.split_array) == id_val_value)
    if args.dry_run:
        train_indices, val_indices, id_val_indices = train_indices[:256], val_indices[:128], id_val_indices[:128]
    settings = config["training"]
    train_loader = DataLoader(IndexedSplit(train_indices), batch_size=int(settings["batch_size"]), shuffle=True, num_workers=int(settings["num_workers"]))
    val_loader = DataLoader(IndexedSplit(val_indices), batch_size=int(settings["batch_size"]), shuffle=False, num_workers=int(settings["num_workers"]))
    model = build_model(config)
    history = train(model, train_loader, val_loader, config, output, args.device, args.dry_run)
    best_state = torch.load(output / "best.ckpt", map_location=args.device); model.load_state_dict(best_state["model"])
    _, id_predictions = evaluate(model, DataLoader(IndexedSplit(id_val_indices), batch_size=int(settings["batch_size"]), shuffle=False, num_workers=int(settings["num_workers"])), args.device, True, "ID validation predictions")
    from medstyleaudit.utils.io import save_table
    id_predictions["split"] = "id_val"; save_table(id_predictions, output / "id_validation_predictions.csv")
    run.complete(status="completed", epochs_completed=len(history), best_checkpoint=str((output / "best.ckpt").resolve()))


if __name__ == "__main__": main()
