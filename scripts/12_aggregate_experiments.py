"""Phase 12: aggregate all prespecified seed/backbone audit runs."""

from pathlib import Path

import json
import pandas as pd

from medstyleaudit.audit.directed import expected_pairs_for_split
from medstyleaudit.audit.experiment_aggregation import aggregate_experiment_runs
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.paths import experiment_path
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.protocol import git_state
from medstyleaudit.utils.io import read_table, save_table


def main() -> None:
    parser = common_parser("Aggregate configured primary-audit runs", "configs/audit/primary.yaml")
    parser.add_argument("--audit-root", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--model-configs", nargs="+", default=["configs/models/resnet50_hf.yaml", "configs/models/densenet121_hf.yaml"])
    parser.add_argument("--coverage-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    split = args.split or config["audit"].get("split", "val")
    models = [load_config(path) for path in args.model_configs]
    expected_runs = {
        model["model"]["architecture"]: {"seeds": model["seeds"], "dataset_version": model["data"]["version"]}
        for model in models
    }
    audit_root = args.audit_root or experiment_path("audits/primary")
    output = args.output_dir or experiment_path("aggregate/primary/by_split") / split
    coverage_dir = args.coverage_dir or experiment_path("p0/matching")
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    run = start_run("aggregate_primary_experiments", config, output, seed, overwrite=args.overwrite)
    hash_path = experiment_path("protocol/protocol_sha256.txt")
    protocol_hash = hash_path.read_text(encoding="utf-8").strip() if hash_path.is_file() else None
    commit, _ = git_state()
    outputs = aggregate_experiment_runs(
        audit_root,
        output,
        expected_runs,
        split=split,
        audit_config=config["audit"],
        expected_pairs=expected_pairs_for_split(config, split),
        coverage_dir=coverage_dir,
        protocol_hash=protocol_hash,
        git_commit=commit,
    )
    primary = experiment_path("aggregate/primary")
    for table in ["directed_hcs", "directed_hce", "global_hcs", "global_hce", "coverage", "seed_stability"]:
        frames = []
        for candidate_split in ("val", "test"):
            path = primary / "by_split" / candidate_split / f"{table}.csv"
            if path.is_file(): frames.append(read_table(path))
        if frames: save_table(pd.concat(frames, ignore_index=True), primary / f"{table}.csv")
    final = experiment_path("aggregate/final"); final.mkdir(parents=True, exist_ok=True)
    main_frames = [frame for name in ("global_hcs", "global_hce") if (frame := primary / f"{name}.csv").is_file()]
    if main_frames: save_table(pd.concat([read_table(path) for path in main_frames], ignore_index=True, sort=False), final / "MAIN_RESULTS.csv")
    matching_frames = [path for path in [experiment_path("p0/matching/matching_balance.csv"), experiment_path("p0/matching/feature_balance.csv"), experiment_path("p0/matching/directed_coverage.csv")] if path.is_file()]
    if matching_frames:
        save_table(pd.concat([read_table(path).assign(source_table=path.stem, number_expected_runs=1, number_completed_runs=1, missing_runs_detail="[]", status="complete", protocol_hash=protocol_hash, git_commit=commit) for path in matching_frames], ignore_index=True, sort=False), final / "MATCHING_RESULTS.csv")

    backbones, seeds = list(expected_runs), [int(seed) for seed in next(iter(expected_runs.values()))["seeds"]]
    audit_dir = experiment_path("audits")
    control_expected = []
    for backbone in backbones:
        for run_seed in seeds:
            control_expected.extend([
                audit_dir / "roi_only/val" / backbone / f"seed_{run_seed:04d}/directed_hcs.csv",
                audit_dir / "context_randomized" / backbone / f"seed_{run_seed:04d}/directed_hcs.csv",
            ])
            control_expected.extend(audit_dir / "planted_shortcut" / backbone / f"seed_{run_seed:04d}" / f"rho_{rho:.2f}/directed_hcs.csv" for rho in (0., .25, .5, .75, 1.))
    robustness_expected = []
    for backbone in backbones:
        for run_seed in seeds:
            robustness_expected.extend([
                audit_dir / "source_buffer" / backbone / f"seed_{run_seed:04d}/robustness_summary.csv",
                audit_dir / "seam" / backbone / f"seed_{run_seed:04d}/robustness_summary.csv",
            ])
            robustness_expected.extend(audit_dir / "identification_ladder" / f"level_{level}/val" / backbone / f"seed_{run_seed:04d}/directed_hcs.csv" for level in range(1, 5))
    lesion_status = audit_dir / "lesion_aware/status.json"
    if lesion_status.is_file() and __import__("json").loads(lesion_status.read_text()).get("status") == "completed":
        robustness_expected.extend(audit_dir / "lesion_aware/val" / backbone / f"seed_{run_seed:04d}/directed_hcs.csv" for backbone in backbones for run_seed in seeds)

    def final_table(paths: list[Path], destination: Path) -> None:
        missing_paths = [str(path) for path in paths if not path.is_file()]
        frames = [read_table(path).assign(source_table=str(path.relative_to(audit_dir))) for path in paths if path.is_file()]
        metadata = {"number_expected_runs": len(paths), "number_completed_runs": len(paths) - len(missing_paths), "missing_runs_detail": json.dumps(missing_paths), "status": "complete" if not missing_paths else "incomplete", "protocol_hash": protocol_hash, "git_commit": commit}
        table = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame([metadata])
        for key, value in metadata.items(): table[key] = value
        save_table(table, destination)

    final_table(control_expected, final / "CONTROL_RESULTS.csv")
    final_table(robustness_expected, final / "ROBUSTNESS_RESULTS.csv")
    missing = int(outputs["missing_runs"]["status"].isin(["missing", "invalid"]).sum())
    run.complete(status="completed" if missing == 0 else "incomplete", missing_runs=missing, expected_runs=sum(len(value["seeds"]) for value in expected_runs.values()))


if __name__ == "__main__":
    main()
