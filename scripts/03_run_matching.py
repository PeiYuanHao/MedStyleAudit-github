"""Phase 3: run balanced matched-triplet selection without model outputs."""

from pathlib import Path

from medstyleaudit.matching.balance import balance_summary, donor_reuse_detail, donor_reuse_distribution, feature_balance, slide_reuse_detail, slide_reuse_distribution
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
    parser.add_argument("--prior-triplets", type=Path, default=None, help="Locked earlier triplets whose donor reuse must carry forward")
    args = parser.parse_args(); config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42)); output = args.output_dir or configured_output(config, "matching")
    settings = config["matching"]
    final_test_requested = args.split in {"test", "ood_test"} or (args.split is None and args.allow_final_test)
    if final_test_requested:
        from medstyleaudit.protocol import assert_frozen_execution_config
        assert_frozen_execution_config(matching=settings)
        enforce_final_test_guard("test", args.allow_final_test)
    run = start_run("balanced_matching", config, output, seed, overwrite=args.overwrite)
    frame = read_table(config["data"]["descriptors"])
    if settings.get("require_validated_lesion_mapping"):
        import json
        from medstyleaudit.utils.paths import experiment_path
        alignment_path = experiment_path("p0/data_integrity/alignment_report.json")
        lesion_path = experiment_path("p0/data_integrity/lesion_features.parquet")
        alignment = json.loads(alignment_path.read_text(encoding="utf-8")) if alignment_path.is_file() else {}
        if str(alignment.get("status", "")).lower() not in {"pass", "passed", "validated", "available"} or not lesion_path.is_file():
            save_json({"status": "unavailable", "reason": "lesion annotation alignment was not validated"}, output / "status.json")
            run.complete(status="unavailable", reason="lesion annotation alignment was not validated")
            return
        lesion = read_table(lesion_path)
        frame = frame.merge(lesion, on="source_id", how="inner", validate="one_to_one")
        settings = dict(settings)
        settings["descriptor_columns"] = list(settings["descriptor_columns"]) + list(settings.get("lesion_columns", []))
    source_frame = frame
    if args.split:
        source_frame = frame[frame["split"] == args.split]
    elif not args.allow_final_test:
        source_frame = frame[frame["split"] != "test"]
    if args.dry_run:
        source_frame = source_frame.groupby(["split", "hospital_id", "label"], group_keys=False).head(16)
        bank_ids = set(source_frame["source_id"])
        bank_frame = frame[(frame["source_id"].isin(bank_ids)) | (frame["split"] == "train")].groupby(["split", "hospital_id", "label"], group_keys=False).head(32)
    else:
        bank_frame = frame
    bank = CandidateBank.build(bank_frame, settings["descriptor_columns"])
    matcher = BalancedTripletMatcher(bank, settings)
    if args.prior_triplets:
        prior = read_table(args.prior_triplets)
        for row in prior.itertuples(index=False):
            for donor, slide in ((row.within_donor, row.within_slide), (row.cross_donor, row.cross_slide)):
                matcher.donor_reuse[donor] += 1; matcher.donor_slide_reuse[slide] += 1
        matcher.saturated_donor_ids = {donor for donor, count in matcher.donor_reuse.items() if count >= int(settings["donor_reuse_cap"])}
        matcher.saturated_slide_ids = {slide for slide, count in matcher.donor_slide_reuse.items() if count >= int(settings.get("donor_slide_reuse_cap", 2**31 - 1))}
    source_rows = bank.frame[bank.frame["source_id"].isin(source_frame["source_id"])]
    capacity = matcher.reuse_capacity_report(source_rows, settings["target_hospitals"])
    save_table(capacity, output / "reuse_capacity.csv")
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
    save_table(triplets, output / "triplets.parquet")
    save_table(ledger, output / "source_ledger.parquet")
    save_table(directed_coverage(ledger), output / "directed_coverage.csv")
    save_table(common_support_coverage(ledger), output / "common_support_coverage.csv")
    save_table(attrition_table(ledger), output / "attrition.csv")
    save_table(balance_summary(triplets) if not triplets.empty else triplets, output / "matching_balance.csv")
    if not triplets.empty:
        save_table(feature_balance(triplets, frame, list(settings["descriptor_columns"])), output / "feature_balance.csv")
        save_table(donor_reuse_detail(triplets, frame), output / "donor_reuse.csv")
        save_table(donor_reuse_distribution(triplets, frame), output / "donor_reuse_summary.csv")
        save_table(slide_reuse_detail(triplets, frame), output / "slide_reuse.csv")
        save_table(slide_reuse_distribution(triplets, frame), output / "slide_reuse_summary.csv")
    save_json(bank.standardizer.to_dict(), output / "descriptor_standardizer.json")
    run.complete(status="completed", accepted_triplets=len(triplets), candidate_comparisons=len(ledger))


if __name__ == "__main__": main()
