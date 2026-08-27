"""Phase 4: regenerate the pre-training matching feasibility gate."""

from pathlib import Path

from medstyleaudit.matching.coverage import attrition_table, common_support_coverage, directed_coverage
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_json, save_table
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.paths import configured_output, experiment_path


def main() -> None:
    args = common_parser("Review matching coverage before GPU training", "configs/matching/primary.yaml").parse_args(); config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42)); output = args.output_dir or experiment_path("matching/coverage_review")
    run = start_run("matching_coverage", config, output, seed, overwrite=args.overwrite)
    ledger = read_table(configured_output(config, "matching") / "coverage" / "source_ledger.csv")
    directed, common = directed_coverage(ledger), common_support_coverage(ledger)
    save_table(directed, output / "directed_coverage.csv"); save_table(common, output / "common_support_coverage.csv"); save_table(attrition_table(ledger), output / "attrition.csv")
    report = {"status": "review_required", "minimum_directed_coverage": float(directed["coverage"].min()), "minimum_common_support_coverage": float(common["coverage"].min()), "message": "Coverage thresholds are scientific decisions; GPU training is not automatically authorized."}
    save_json(report, output / "feasibility_gate.json"); run.complete(status="completed", gate_status="review_required")


if __name__ == "__main__": main()
