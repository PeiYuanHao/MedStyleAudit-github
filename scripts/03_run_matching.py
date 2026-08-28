"""Phase 3: run balanced matched-triplet selection without model outputs."""

from pathlib import Path

from medstyleaudit.matching.balance import balance_summary
from medstyleaudit.matching.candidate_bank import CandidateBank
from medstyleaudit.matching.coverage import annotate_common_support, attrition_table, common_support_coverage, directed_coverage
from medstyleaudit.matching.triplet_matcher import BalancedTripletMatcher
from medstyleaudit.utils.cli import common_parser, enforce_final_test_guard
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_json, save_table
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.paths import configured_output


def main() -> None:
    parser = common_parser("Run balanced matched-triplet donor selection", "configs/matching/primary.yaml")
    parser.add_argument("--split", default=None)
    args = parser.parse_args(); config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42)); output = args.output_dir or configured_output(config, "matching")
    run = start_run("balanced_matching", config, output, seed, overwrite=args.overwrite)
    frame = read_table(config["data"]["descriptors"])
    source_frame = frame
    if args.split:
        enforce_final_test_guard(args.split, args.allow_final_test); source_frame = frame[frame["split"] == args.split]
    elif not args.allow_final_test:
        source_frame = frame[frame["split"] != "test"]
    if args.dry_run:
        source_frame = source_frame.groupby(["split", "hospital_id", "label"], group_keys=False).head(16)
        bank_ids = set(source_frame["source_id"])
        bank_frame = frame[(frame["source_id"].isin(bank_ids)) | (frame["split"] == "train")].groupby(["split", "hospital_id", "label"], group_keys=False).head(32)
    else:
        bank_frame = frame
    settings = config["matching"]
    bank = CandidateBank.build(bank_frame, settings["descriptor_columns"])
    matcher = BalancedTripletMatcher(bank, settings)
    source_rows = bank.frame[bank.frame["source_id"].isin(source_frame["source_id"])]
    capacity = matcher.reuse_capacity_report(source_rows, settings["target_hospitals"])
    save_table(capacity, output / "coverage" / "reuse_capacity.csv")
    infeasible = capacity[~capacity["feasible"]]
    if not infeasible.empty:
        run.complete(status="unavailable", reason="infeasible_reuse_capacity", infeasible_scopes=len(infeasible))
        raise RuntimeError(
            "Configured donor reuse caps are infeasible; see coverage/reuse_capacity.csv. "
            "Matching was stopped before the source loop."
        )
    triplets, ledger = matcher.match(source_rows, settings["target_hospitals"], show_progress=True)
    ledger = annotate_common_support(ledger)
    if not triplets.empty:
        common_lookup = ledger.set_index(["source_split", "source_hospital", "target_hospital", "source_id"])["common_support"]
        triplets["common_support"] = [common_lookup.loc[(row.source_split, row.source_hospital, row.target_hospital, row.source_id)] for row in triplets.itertuples()]
    save_table(triplets, output / "triplets" / "triplets.csv")
    save_table(ledger, output / "coverage" / "source_ledger.csv")
    save_table(directed_coverage(ledger), output / "coverage" / "directed_coverage.csv")
    save_table(common_support_coverage(ledger), output / "coverage" / "common_support_coverage.csv")
    save_table(attrition_table(ledger), output / "coverage" / "attrition.csv")
    save_table(balance_summary(triplets) if not triplets.empty else triplets, output / "balance" / "matching_balance.csv")
    save_json(bank.standardizer.to_dict(), output / "descriptor_standardizer.json")
    run.complete(status="completed", accepted_triplets=len(triplets), candidate_comparisons=len(ledger))


if __name__ == "__main__": main()
