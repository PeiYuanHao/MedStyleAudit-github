"""Phase 8: construct the locked source-buffer/seam robustness grid."""

from pathlib import Path

import pandas as pd

from medstyleaudit.counterfactual.source_buffer import intervention_area_ratio
from medstyleaudit.audit.directed import aggregate_audit
from medstyleaudit.audit.inference import infer_triplets
from medstyleaudit.data.wilds_loader import load_wilds_dataset
from medstyleaudit.models.trainer import build_model
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_table
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.paths import configured_output, experiment_path


def main() -> None:
    parser = common_parser("Run robustness grid using locked triplets", "configs/audit/source_buffer.yaml"); parser.add_argument("--triplets", type=Path, default=None); parser.add_argument("--checkpoint", type=Path, default=None); parser.add_argument("--model-config", default="configs/models/resnet50.yaml"); parser.add_argument("--id-logits", type=Path, default=None); parser.add_argument("--split", default="val")
    args = parser.parse_args(); config = load_config(args.config); seed = args.seed if args.seed is not None else int(config.get("seed", 42)); output = args.output_dir or configured_output(config, "robustness")
    triplet_path = args.triplets or experiment_path("matching/triplets/triplets.csv")
    run = start_run(config.get("experiment", "robustness"), config, output, seed, overwrite=args.overwrite)
    triplets = read_table(triplet_path); triplets = triplets[triplets["source_split"] == args.split]
    if args.split == "test" and not args.allow_final_test: raise PermissionError("Final hospital-2 evaluation requires --allow-final-test")
    if args.dry_run: triplets = triplets.head(8)
    settings = config["audit"]; rows = []
    buffers = settings.get("source_buffers", [0]); modes = settings.get("boundary_modes", ["feathered"]); widths = settings.get("feather_widths", [settings.get("feather_width", 4)])
    for buffer in buffers:
        for mode in modes:
            for width in widths:
                effective = 0 if mode == "hard" else int(width)
                rows.append({"source_buffer": int(buffer), "intervention_area_ratio": intervention_area_ratio((96, 96), 32, int(buffer)), "boundary_mode": mode, "feather_width": effective, "triplet_file": str(triplet_path), "n_locked_triplets": len(triplets), "status": "pending_inference"})
    grid = pd.DataFrame(rows).drop_duplicates(); save_table(grid, output / "robustness_grid.csv")
    if args.checkpoint:
        import torch
        from torchvision.transforms import Compose, Normalize, ToTensor
        model_config = load_config(args.model_config); model = build_model(model_config)
        state = torch.load(args.checkpoint, map_location=args.device); model.load_state_dict(state["model"])
        transform = Compose([ToTensor(), Normalize(model_config["model"]["input_mean"], model_config["model"]["input_std"])])
        dataset = load_wilds_dataset({"data": {**model_config["data"], "download": False}})
        if not args.id_logits: raise ValueError("--id-logits is required when --checkpoint is supplied")
        id_table = read_table(args.id_logits); logit_column = "logit" if "logit" in id_table else "original_logit"
        all_sources = []
        for setting_index, setting in grid.iterrows():
            setting_dir = output / f"setting_{setting_index:03d}"
            predictions, qa = infer_triplets(model, dataset, triplets, transform, device=args.device, source_buffer=int(setting.source_buffer), feather_width=int(setting.feather_width), show_progress=True)
            predictions["seed"] = seed; predictions["backbone"] = model_config["model"]["architecture"]
            save_table(predictions, setting_dir / "predictions.csv"); save_table(qa, setting_dir / "construction_qa.csv")
            tables = aggregate_audit(predictions, id_table[logit_column], setting_dir, float(config.get("audit", {}).get("q_min", .001)), int(config.get("audit", {}).get("bootstrap_draws", 0)), seed)
            source_table = tables["source_metrics"].copy()
            source_table["source_buffer"] = int(setting.source_buffer); source_table["boundary_mode"] = setting.boundary_mode; source_table["feather_width"] = int(setting.feather_width)
            all_sources.append(source_table)
        from medstyleaudit.audit.robustness import robustness_summary
        combined = pd.concat(all_sources, ignore_index=True); save_table(combined, output / "source_metrics_all_settings.csv"); save_table(robustness_summary(combined, ["source_buffer", "boundary_mode", "feather_width"]), output / "robustness_summary.csv")
        grid["status"] = "completed"; save_table(grid, output / "robustness_grid.csv")
    run.complete(status="completed", n_grid_settings=len(grid), inference_completed=bool(args.checkpoint))


if __name__ == "__main__": main()
