"""Phase 9: generate donor ledgers for the prespecified identification ladder."""

import pandas as pd

from medstyleaudit.matching.candidate_bank import CandidateBank
from medstyleaudit.matching.ladder import advanced_level_status, independent_nearest_pairs, random_pairs
from medstyleaudit.utils.cli import common_parser, enforce_final_test_guard
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_table
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.paths import configured_output, experiment_path


def main() -> None:
    parser = common_parser("Build identification-ladder donor policies", "configs/matching/primary.yaml"); parser.add_argument("--level", type=int, choices=range(1, 7), required=True); parser.add_argument("--target-hospital", type=int, required=True); parser.add_argument("--split", default="val")
    args = parser.parse_args(); config = load_config(args.config); seed = args.seed if args.seed is not None else int(config.get("seed", 42)); output = args.output_dir or experiment_path("identification_ladder") / f"level_{args.level}"
    run = start_run(f"identification_ladder_level_{args.level}", config, output, seed, overwrite=args.overwrite)
    enforce_final_test_guard(args.split, args.allow_final_test)
    frame = read_table(config["data"]["descriptors"]); settings = config["matching"]; bank = CandidateBank.build(frame, settings["descriptor_columns"]); sources = bank.frame[(bank.frame["split"] == args.split) & (bank.frame["hospital_id"] != args.target_hospital)]
    if args.dry_run: sources = sources.head(32)
    if args.level == 2: triplets = random_pairs(bank, sources, args.target_hospital, int(settings["donors_per_source"]), seed)
    elif args.level == 3: triplets = independent_nearest_pairs(bank, sources, args.target_hospital, int(settings["donors_per_source"]))
    elif args.level == 4:
        primary = configured_output(config, "p0/matching") / "triplets.parquet"; triplets = read_table(primary); triplets = triplets[(triplets["target_hospital"] == args.target_hospital) & (triplets["source_split"] == args.split)]
    elif args.level in {5, 6}:
        required = settings.get("lesion_columns" if args.level == 5 else "appearance_columns", [])
        status = advanced_level_status(args.level, frame.columns, required)
        save_table(pd.DataFrame([status]), output / "status.csv")
        run.complete(status=status["status"], analysis_status=status["status"], reason=status["reason"])
        return
    else:
        triplets = random_pairs(bank, sources, args.target_hospital, 1, seed)
        triplets["within_donor"] = triplets["source_id"]
    if not triplets.empty:
        lookup = bank.frame.drop_duplicates("source_id").set_index("source_id")
        triplets = triplets.reset_index(drop=True)
        triplets["triplet_id"] = [f"l{args.level}-h{args.target_hospital}-t{index:09d}" for index in range(len(triplets))]
        for prefix, id_column in (("source", "source_id"), ("within", "within_donor"), ("cross", "cross_donor")):
            triplets[f"{prefix}_slide"] = triplets[id_column].map(lookup["slide_id"])
            triplets[f"{prefix}_physical_id"] = triplets[id_column].map(lookup["physical_id"])
        triplets["common_support"] = True
    save_table(triplets, output / "triplets.parquet"); save_table(pd.DataFrame([{"level": args.level, "candidate_sources": len(sources), "accepted_sources": triplets["source_id"].nunique() if not triplets.empty else 0, "coverage": triplets["source_id"].nunique() / max(len(sources), 1)}]), output / "coverage.csv"); run.complete(status="completed", accepted_rows=len(triplets))


if __name__ == "__main__": main()
