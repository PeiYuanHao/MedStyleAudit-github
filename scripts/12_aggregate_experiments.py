"""Phase 12: aggregate the reduced final-suite audit matrix.

Expected model is ResNet-50 only, seeds 11/42/101, and the two held-out audit
populations hospital1 (validation) and hospital2 (final test). Aggregation never
silently averages a partial seed set: if any seed is missing the population
status is ``incomplete``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from medstyleaudit.audit.directed import expected_pairs_for_split
from medstyleaudit.audit.experiment_aggregation import aggregate_experiment_runs
from medstyleaudit.protocol import git_state
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_table
from medstyleaudit.utils.paths import experiment_path
from medstyleaudit.utils.run_metadata import start_run

BACKBONE = "resnet50"
PRIMARY_TABLE_SETTINGS = ("primary", "random_paired", "roi_only")
ROBUSTNESS_SETTINGS = ("buffer_r8", "hard_boundary")

_RUN_TABLE_FILES = (
    "directed_hcs.csv",
    "directed_hce.csv",
    "global_hcs_pair_weighted.csv",
    "global_hcs_common_support.csv",
    "global_hce_pair_weighted.csv",
    "global_hce_common_support.csv",
)


def _seed_dir(population: str, setting: str, seed: int) -> Path:
    return experiment_path("audits") / population / setting / BACKBONE / f"seed_{seed:04d}"


def _settings_completeness(population: str, seeds: list[int]) -> pd.DataFrame:
    rows = []
    settings = list(PRIMARY_TABLE_SETTINGS) + list(ROBUSTNESS_SETTINGS)
    for setting in settings:
        expected = len(seeds)
        completed = 0
        missing = []
        for seed in seeds:
            directory = _seed_dir(population, setting, seed)
            if setting in ROBUSTNESS_SETTINGS:
                present = (directory / "robustness_summary.csv").is_file()
            else:
                present = all((directory / name).is_file() for name in ("directed_hcs.csv", "directed_hce.csv"))
            if present:
                completed += 1
            else:
                missing.append({"backbone": BACKBONE, "seed": seed, "setting": setting, "run_path": str(directory)})
        rows.append({
            "population": population, "setting": setting, "backbone": BACKBONE,
            "expected_runs": expected, "completed_runs": completed,
            "missing_runs": json.dumps(missing), "status": "complete" if completed == expected else "incomplete",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = common_parser("Aggregate the reduced final audit matrix", "configs/audit/primary.yaml")
    parser.add_argument("--population", choices=["hospital1", "hospital2"], default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    split = args.split or ("val" if args.population == "hospital1" else "test")
    population = args.population or ("hospital1" if split == "val" else "hospital2")
    if args.split == "test" or population == "hospital2":
        from medstyleaudit.utils.cli import enforce_final_test_guard
        enforce_final_test_guard(split, args.allow_final_test)
    model_config = load_config("configs/models/resnet50_hf.yaml")
    seeds = [int(seed) for seed in model_config["seeds"]]
    expected_runs = {BACKBONE: {"seeds": seeds, "dataset_version": model_config["data"]["version"]}}
    output = args.output_dir or experiment_path("aggregate") / population
    coverage_dir = experiment_path("p0/matching")
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    run = start_run("aggregate_final_suite", config, output, seed, overwrite=args.overwrite)
    hash_path = experiment_path("protocol/protocol_sha256.txt")
    protocol_hash = hash_path.read_text(encoding="utf-8").strip() if hash_path.is_file() else None
    commit, _ = git_state()
    expected_pairs = expected_pairs_for_split(config, split)
    audit_root = experiment_path("audits") / population

    global_hcs, global_hce, global_intervals, directed_intervals, individual_seeds = [], [], [], [], []
    primary_outputs = {}
    for setting in PRIMARY_TABLE_SETTINGS:
        setting_output = output / setting
        primary_outputs[setting] = aggregate_experiment_runs(
            audit_root / setting,
            setting_output,
            expected_runs,
            split=split,
            audit_config=config["audit"],
            expected_pairs=expected_pairs,
            coverage_dir=coverage_dir,
            protocol_hash=protocol_hash,
            git_commit=commit,
        )
        for frame in primary_outputs[setting]["combined_global_hcs"].assign(setting=setting, population=population).to_dict("records"):
            global_hcs.append(frame)
        for frame in primary_outputs[setting]["combined_global_hce"].assign(setting=setting, population=population).to_dict("records"):
            global_hce.append(frame)
        if not primary_outputs[setting]["combined_global_intervals"].empty:
            global_intervals.append(primary_outputs[setting]["combined_global_intervals"].assign(setting=setting, population=population))
        if not primary_outputs[setting]["combined_directed_intervals"].empty:
            directed_intervals.append(primary_outputs[setting]["combined_directed_intervals"].assign(setting=setting, population=population))
        seed_table = pd.concat(
            [
                primary_outputs[setting]["combined_directed_hcs"].assign(metric="directed_hcs", setting=setting, population=population),
                primary_outputs[setting]["combined_directed_hce"].assign(metric="directed_hce", setting=setting, population=population),
            ],
            ignore_index=True, sort=False,
        ) if not primary_outputs[setting]["combined_directed_hcs"].empty else pd.DataFrame()
        if not seed_table.empty:
            individual_seeds.append(seed_table)

    save_table(pd.DataFrame(global_hcs) if global_hcs else pd.DataFrame(), output / "global_hcs.csv")
    save_table(pd.DataFrame(global_hce) if global_hce else pd.DataFrame(), output / "global_hce.csv")
    save_table(pd.concat(global_intervals, ignore_index=True, sort=False) if global_intervals else pd.DataFrame(), output / "global_intervals.csv")
    save_table(pd.concat(directed_intervals, ignore_index=True, sort=False) if directed_intervals else pd.DataFrame(), output / "directed_intervals.csv")
    if individual_seeds:
        save_table(pd.concat(individual_seeds, ignore_index=True, sort=False), output / "INDIVIDUAL_SEEDS.csv")

    robustness_rows = []
    for setting in ROBUSTNESS_SETTINGS:
        for run_seed in seeds:
            directory = _seed_dir(population, setting, run_seed)
            summary = directory / "robustness_summary.csv"
            if summary.is_file():
                robustness_rows.append(read_table(summary).assign(setting=setting, population=population, seed=run_seed, backbone=BACKBONE))
    if robustness_rows:
        save_table(pd.concat(robustness_rows, ignore_index=True, sort=False), output / "robustness_by_seed.csv")

    completeness = _settings_completeness(population, seeds)
    save_table(completeness, output / "run_completeness.csv")

    matching_frames = [
        path for path in [
            experiment_path("p0/matching/matching_balance.csv"),
            experiment_path("p0/matching/feature_balance.csv"),
            experiment_path("p0/matching/directed_coverage.csv"),
            experiment_path("p0/matching/common_support_coverage.csv"),
        ] if path.is_file()
    ]
    if matching_frames:
        matching = pd.concat(
            [read_table(path).assign(source_table=path.stem, population=population, protocol_hash=protocol_hash, git_commit=commit) for path in matching_frames],
            ignore_index=True, sort=False,
        )
        save_table(matching, output / "matching_results.csv")

    run.complete(
        status=("completed" if (completeness["status"] == "complete").all() else "incomplete"),
        expected_runs=int(completeness["expected_runs"].sum()),
        completed_runs=int(completeness["completed_runs"].sum()),
    )


if __name__ == "__main__":
    main()
