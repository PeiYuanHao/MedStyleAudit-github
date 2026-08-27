"""Phase 6: frozen-model primary HCS/HCE audit."""

from pathlib import Path

import pandas as pd

from medstyleaudit.audit.directed import aggregate_audit, expected_pairs_for_split
from medstyleaudit.audit.inference import infer_triplets, ROIIdentityError
from medstyleaudit.data.wilds_loader import load_wilds_dataset
from medstyleaudit.models.trainer import build_model
from medstyleaudit.utils.cli import common_parser, enforce_final_test_guard
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_json, save_table
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.paths import configured_output, experiment_path


def main() -> None:
    parser = common_parser("Run the primary paired context audit", "configs/audit/primary.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--model-config", default="configs/models/resnet50.yaml"); parser.add_argument("--triplets", type=Path, default=None); parser.add_argument("--id-logits", type=Path, default=None); parser.add_argument("--split", default=None); parser.add_argument("--roi-only-control", action="store_true")
    args = parser.parse_args(); config = load_config(args.config); model_config = load_config(args.model_config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42)); split = args.split or config["audit"].get("split", "val"); enforce_final_test_guard(split, args.allow_final_test)
    architecture = model_config["model"]["architecture"]
    output_root = experiment_path("controls/roi_only") if args.roi_only_control else configured_output(config, "primary_audit")
    output = args.output_dir or output_root / architecture / f"seed_{seed:04d}" / split
    config["checkpoint"] = str(args.checkpoint)
    config.setdefault("data", {})["version"] = model_config.get("data", {}).get("version")
    config["model_architecture"] = architecture
    run = start_run("primary_audit", config, output, seed, overwrite=args.overwrite)
    triplets = read_table(args.triplets or config["data"]["triplets"]); triplets = triplets[triplets["source_split"] == split]
    if args.dry_run: triplets = triplets.head(8)
    if triplets.empty: raise RuntimeError(f"No accepted triplets for split={split}")
    import torch
    from torchvision.transforms import Compose, Normalize, ToTensor
    model = build_model(model_config); state = torch.load(args.checkpoint, map_location=args.device); model.load_state_dict(state["model"])
    transform = Compose([ToTensor(), Normalize(model_config["model"]["input_mean"], model_config["model"]["input_std"])])
    dataset_config = {"data": {**model_config["data"], "download": False}}
    try:
        predictions, qa = infer_triplets(model, load_wilds_dataset(dataset_config), triplets, transform, device=args.device, roi_size=int(config["audit"]["roi_size"]), source_buffer=int(config["audit"]["source_buffer"]), feather_width=int(config["audit"]["feather_width"]), roi_only_control=args.roi_only_control)
    except ROIIdentityError as error:
        save_table(error.qa, output / "construction_qa.csv")
        save_json({"status": "failed", "reason": str(error)}, output / "roi_qa_failure.json")
        run.complete(status="failed", roi_identity="FAIL", reason=str(error))
        raise
    predictions["seed"], predictions["backbone"] = seed, architecture
    save_table(predictions, output / "predictions.csv"); save_table(qa, output / "construction_qa.csv")
    id_path = args.id_logits or Path(config["data"]["id_validation_logits"]); id_table = read_table(id_path)
    logit_column = "logit" if "logit" in id_table else "original_logit"
    save_json({"checkpoint": str(args.checkpoint), "split": split, "id_logit_file": str(id_path), "id_logit_column": logit_column}, output / "audit_inputs.json")
    aggregate_audit(predictions, id_table[logit_column], output, float(config["audit"]["q_min"]), int(config["audit"].get("bootstrap_draws", 0)), seed, expected_pairs_for_split(config, split))
    run.complete(status="completed", n_prediction_rows=len(predictions), roi_identity="PASS")


if __name__ == "__main__": main()
