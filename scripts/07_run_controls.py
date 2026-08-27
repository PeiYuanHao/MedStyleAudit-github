"""Phase 7: audit validity controls and planted-shortcut data generation."""

from pathlib import Path

import numpy as np
import pandas as pd

from medstyleaudit.controls.context_randomized import joint_cell_quotas
from medstyleaudit.controls.permutation import permute_assignments
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_table
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.paths import configured_output, experiment_path


def main() -> None:
    parser = common_parser("Run a configured audit validity control", "configs/controls/roi_only.yaml")
    parser.add_argument("--descriptor-table", type=Path, default=None); parser.add_argument("--assignments", type=Path, default=None)
    args = parser.parse_args(); config = load_config(args.config); experiment = config.get("experiment", "control")
    seed = args.seed if args.seed is not None else int(config.get("seed", 42)); output = args.output_dir or configured_output(config, f"controls/{experiment}")
    descriptor_table = args.descriptor_table or experiment_path("descriptors/descriptors.csv")
    run = start_run(experiment, config, output, seed, overwrite=args.overwrite)
    if experiment == "context_randomized_control":
        donors = read_table(descriptor_table); quotas = joint_cell_quotas(donors[donors["split"] == "train"], min(len(donors), 128) if args.dry_run else len(donors[donors["split"] == "train"]))
        if (quotas["quota"] > quotas["available"] * int(config["control"]["donor_reuse_cap"])).any(): raise RuntimeError("Fixed context-randomized cell quota is infeasible")
        save_table(quotas, output / "assignment_quotas.csv")
    elif experiment == "planted_shortcut_control" and args.assignments:
        table = read_table(args.assignments); table["permuted_cue"] = permute_assignments(table["cue"].to_numpy(), seed, table["hospital_id"].to_numpy() if "hospital_id" in table else None); save_table(table, output / "permuted_assignments.csv")
    else:
        pd.DataFrame([{"control": experiment, "status": "configured", "message": "Use this config with training/audit entry points; no empirical result was generated."}]).to_csv(output / "control_status.csv", index=False)
    run.complete(status="completed", control_status="configured")


if __name__ == "__main__": main()
