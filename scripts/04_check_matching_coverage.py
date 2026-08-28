"""Phase 4: regenerate the pre-training matching feasibility gate."""

from medstyleaudit.matching.coverage import attrition_table, common_support_coverage, directed_coverage
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_json, save_table
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.paths import configured_output, experiment_path


def main() -> None:
    args = common_parser("Review matching coverage before GPU training", "configs/matching/primary.yaml").parse_args(); config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42)); output = args.output_dir or experiment_path("p0/matching/coverage_review")
    run = start_run("matching_coverage", config, output, seed, overwrite=args.overwrite)
    matching = configured_output(config, "p0/matching")
    ledger = read_table(matching / "source_ledger.parquet")
    directed, common = directed_coverage(ledger), common_support_coverage(ledger)
    save_table(directed, output / "directed_coverage.csv"); save_table(common, output / "common_support_coverage.csv"); save_table(attrition_table(ledger), output / "attrition.csv")
    minimum_directed, minimum_common = float(directed["coverage"].min()), float(common["coverage"].min())
    report = {"status": "PASS" if minimum_directed >= .90 and minimum_common >= .90 else "FAIL", "minimum_directed_coverage": minimum_directed, "minimum_common_support_coverage": minimum_common, "threshold": .90}
    save_json(report, output / "coverage_gate.json"); run.complete(status="completed", gate_status=report["status"])
    if report["status"] != "PASS": raise SystemExit("Matching coverage gate FAIL")


if __name__ == "__main__": main()
