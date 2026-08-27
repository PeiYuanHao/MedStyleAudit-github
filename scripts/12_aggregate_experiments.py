"""Phase 12: aggregate all prespecified seed/backbone audit runs."""

from pathlib import Path

from medstyleaudit.audit.directed import expected_pairs_for_split
from medstyleaudit.audit.experiment_aggregation import aggregate_experiment_runs
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.paths import configured_output, experiment_path
from medstyleaudit.utils.run_metadata import start_run


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
    audit_root = args.audit_root or configured_output(config, "primary_audit")
    output = args.output_dir or experiment_path("aggregate") / split
    coverage_dir = args.coverage_dir or experiment_path("matching/coverage_review")
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    run = start_run("aggregate_primary_experiments", config, output, seed, overwrite=args.overwrite)
    outputs = aggregate_experiment_runs(
        audit_root,
        output,
        expected_runs,
        split=split,
        audit_config=config["audit"],
        expected_pairs=expected_pairs_for_split(config, split),
        coverage_dir=coverage_dir,
    )
    missing = len(outputs["missing_runs"])
    run.complete(status="completed" if missing == 0 else "partial", missing_runs=missing, expected_runs=sum(len(value["seeds"]) for value in expected_runs.values()))


if __name__ == "__main__":
    main()
