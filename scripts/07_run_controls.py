"""Run the frozen context-randomized or planted-shortcut control end to end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from medstyleaudit.audit.directed import aggregate_audit, expected_pairs_for_split
from medstyleaudit.audit.inference import infer_triplets
from medstyleaudit.controls.context_randomized import independent_context_assignments, joint_cell_quotas
from medstyleaudit.controls.planted_shortcut import plant_corner_cue
from medstyleaudit.counterfactual.transplant import transplant
from medstyleaudit.data.wilds_loader import load_wilds_dataset
from medstyleaudit.data.loading import loader_kwargs
from medstyleaudit.models.trainer import build_model, evaluate, train
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_table
from medstyleaudit.utils.paths import configured_output, experiment_path
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.seed import seed_everything


def _loaders(dataset, metadata, model_config, settings, seed, mode, ledger=None, rho=0.0, dry_run=False):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import Compose, Normalize, ToTensor

    transform = Compose([ToTensor(), Normalize(model_config["model"]["input_mean"], model_config["model"]["input_std"])])
    train_meta = metadata[metadata["split"] == "train"].sort_values("source_id")
    id_meta = metadata[metadata["split"] == "id_val"].sort_values("source_id")
    val_meta = metadata[metadata["split"] == "val"].sort_values("source_id")
    if dry_run:
        train_meta, id_meta, val_meta = train_meta.head(256), id_meta.head(128), val_meta.head(128)
    donor_lookup = ledger.set_index("source_id")["donor_source_id"].to_dict() if ledger is not None else {}

    class ControlDataset(Dataset):
        def __init__(self, frame, training): self.frame, self.training = frame.reset_index(drop=True), training
        def __len__(self): return len(self.frame)
        def __getitem__(self, index):
            row = self.frame.iloc[index]; image, label, _ = dataset[int(row.source_id)]
            array = np.asarray(image.convert("RGB"))
            if self.training and mode == "context_randomized":
                donor, _, _ = dataset[int(donor_lookup[row.source_id])]
                array = transplant(array, np.asarray(donor.convert("RGB")), roi_size=32, feather_width=4)
            elif self.training and mode == "planted_shortcut":
                local_rng = np.random.default_rng(seed * 1_000_003 + int(row.source_id))
                array = plant_corner_cue(array, int(label), rho, int(settings.get("cue_size", 8)), local_rng)
            return transform(Image.fromarray(array)), torch.as_tensor(label), int(row.source_id)

    kwargs = loader_kwargs(model_config["training"], str(settings.get("device", "cpu")))
    return (
        DataLoader(ControlDataset(train_meta, True), shuffle=True, **kwargs),
        DataLoader(ControlDataset(val_meta, False), shuffle=False, **kwargs),
        DataLoader(ControlDataset(id_meta, False), shuffle=False, **kwargs),
        transform,
    )


def _train_and_audit(dataset, metadata, model_config, audit_config, settings, output, seed, mode, triplets, device, dry_run, ledger=None, rho=0.0):
    import torch
    runtime_settings = {**settings, "device": device}
    train_loader, val_loader, id_loader, transform = _loaders(dataset, metadata, model_config, runtime_settings, seed, mode, ledger, rho, dry_run)
    control_model_config = {**model_config, "training": {**model_config["training"], "epochs": int(settings.get("epochs", model_config["training"]["epochs"]))}}
    last_checkpoint = output / "checkpoint" / "last.ckpt"
    if last_checkpoint.is_file():
        control_model_config["training"]["resume"] = str(last_checkpoint)
    model = build_model(control_model_config)
    train(model, train_loader, val_loader, control_model_config, output / "checkpoint", device, dry_run)
    state = torch.load(output / "checkpoint" / "best.ckpt", map_location=device); model.load_state_dict(state["model"])
    inference_options = {
        "amp": bool(control_model_config["training"].get("amp", False)),
        "channels_last": bool(control_model_config["training"].get("channels_last", False)),
    }
    _, id_predictions = evaluate(model, id_loader, device, True, "control ID validation", **inference_options)
    save_table(id_predictions.assign(split="id_val"), output / "id_val.parquet")
    selected = triplets[triplets["source_split"] == "val"]
    if dry_run: selected = selected.head(8)
    predictions, qa = infer_triplets(
        model,
        dataset,
        selected,
        transform,
        device=device,
        roi_size=32,
        feather_width=4,
        batch_size=int(audit_config["audit"].get("inference_batch_size", 64)),
        show_progress=True,
    )
    predictions["seed"], predictions["backbone"] = seed, model_config["model"]["architecture"]
    save_table(predictions, output / "audit_records.parquet"); save_table(qa, output / "construction_qa.csv")
    tables = aggregate_audit(predictions, id_predictions["logit"], output, float(audit_config["audit"]["q_min"]), int(audit_config["audit"].get("bootstrap_draws", 0)), seed, expected_pairs_for_split(audit_config, "val"))
    unavailable = [name for name in ("global_hcs_pair_weighted", "global_hcs_common_support", "global_hce_pair_weighted", "global_hce_common_support") if "status" in tables[name] and (tables[name]["status"] != "available").any()]
    if unavailable: raise RuntimeError(f"Control audit is missing a required directed hospital pair: {unavailable}")


def main() -> None:
    parser = common_parser("Run a frozen audit validity control", "configs/controls/context_randomized.yaml")
    parser.add_argument("--descriptor-table", type=Path, default=None)
    parser.add_argument("--model-config", default="configs/models/resnet50_hf.yaml")
    parser.add_argument("--audit-config", default="configs/audit/primary.yaml")
    parser.add_argument("--triplets", type=Path, default=None)
    parser.add_argument("--force-rhos", action="store_true", help="Recompute completed planted-shortcut rho values")
    args = parser.parse_args(); config = load_config(args.config); experiment = config.get("experiment", "control")
    if experiment == "roi_only_control":
        raise SystemExit("ROI-only uses scripts/06_run_primary_audit.py --roi-only-control with a primary checkpoint")
    seed = args.seed if args.seed is not None else int(config.get("seed", 42)); seed_everything(seed)
    model_config, audit_config = load_config(args.model_config), load_config(args.audit_config)
    architecture = model_config["model"]["architecture"]
    output_root = args.output_dir or configured_output(config, f"audits/{experiment}")
    output = output_root / architecture / f"seed_{seed:04d}"
    run = start_run(experiment, config, output, seed, overwrite=args.overwrite)
    descriptor_path = args.descriptor_table or experiment_path("p0/descriptors/descriptors.parquet")
    metadata = read_table(descriptor_path); dataset = load_wilds_dataset(model_config)
    triplets = read_table(args.triplets or experiment_path("p0/matching/triplets.parquet"))
    settings = config["control"]
    if experiment == "context_randomized_control":
        sources = metadata[metadata["split"] == "train"]
        if args.dry_run: sources = sources.head(256)
        ledger, diagnostics = independent_context_assignments(sources, sources, seed=seed, donor_reuse_cap=int(settings["donor_reuse_cap"]))
        save_table(ledger, output / "assignment_ledger.parquet"); save_table(diagnostics, output / "assignment_diagnostics.csv")
        save_table(joint_cell_quotas(sources, len(sources)), output / "assignment_quotas.csv")
        _train_and_audit(dataset, metadata, model_config, audit_config, settings, output, seed, "context_randomized", triplets, args.device, args.dry_run, ledger=ledger)
        save_table(pd.DataFrame([{"stage": stage, "status": "completed"} for stage in ["assignment", "training", "audit"]]), output / "control_status.csv")
        run.complete(status="completed", assignment_rows=len(ledger))
    elif experiment == "planted_shortcut_control":
        completed = []
        for rho in map(float, settings["strengths"]):
            rho_output = output / f"rho_{rho:.2f}"
            required = [
                rho_output / "checkpoint" / "best.ckpt",
                rho_output / "id_val.parquet",
                rho_output / "audit_records.parquet",
                rho_output / "directed_hcs.csv",
                rho_output / "directed_hce.csv",
            ]
            if not args.force_rhos and all(path.is_file() for path in required):
                completed.append({"rho": rho, "status": "completed", "resumed": True})
                save_table(pd.DataFrame(completed), output / "rho_status.csv")
                continue
            train_rows = metadata[metadata["split"] == "train"]
            assignments = []
            for row in train_rows.itertuples(index=False):
                rng = np.random.default_rng(seed * 1_000_003 + int(row.source_id))
                assignments.append({"source_id": row.source_id, "label": int(row.label), "rho": rho, "cue_applied": bool(rng.random() <= rho)})
            save_table(pd.DataFrame(assignments), rho_output / "assignment_ledger.parquet")
            _train_and_audit(dataset, metadata, model_config, audit_config, settings, rho_output, seed, "planted_shortcut", triplets, args.device, args.dry_run, rho=rho)
            completed.append({"rho": rho, "status": "completed", "resumed": False})
            save_table(pd.DataFrame(completed), output / "rho_status.csv")
        run.complete(status="completed", completed_rho=len(completed))
    else:
        raise ValueError(f"Unsupported required control: {experiment}")


if __name__ == "__main__":
    main()
