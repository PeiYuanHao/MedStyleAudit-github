"""Phase 9: generate donor ledgers for the prespecified identification ladder."""

from pathlib import Path

import pandas as pd

from medstyleaudit.matching.candidate_bank import CandidateBank
from medstyleaudit.matching.ladder import independent_nearest_pairs, random_pairs
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_table
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.paths import configured_output, experiment_path


def main() -> None:
    parser = common_parser("Build identification-ladder donor policies", "configs/matching/primary.yaml"); parser.add_argument("--level", type=int, choices=range(1, 7), required=True); parser.add_argument("--target-hospital", type=int, required=True)
    args = parser.parse_args(); config = load_config(args.config); seed = args.seed if args.seed is not None else int(config.get("seed", 42)); output = args.output_dir or experiment_path("identification_ladder") / f"level_{args.level}"
    run = start_run(f"identification_ladder_level_{args.level}", config, output, seed, overwrite=args.overwrite)
    frame = read_table(config["data"]["descriptors"]); settings = config["matching"]; bank = CandidateBank.build(frame, settings["descriptor_columns"]); sources = bank.frame[bank.frame["hospital_id"] != args.target_hospital]
    if args.dry_run: sources = sources.head(32)
    if args.level == 2: triplets = random_pairs(bank, sources, args.target_hospital, int(settings["donors_per_source"]), seed)
    elif args.level == 3: triplets = independent_nearest_pairs(bank, sources, args.target_hospital, int(settings["donors_per_source"]))
    elif args.level == 4:
        primary = configured_output(config, "matching") / "triplets" / "triplets.csv"; triplets = read_table(primary); triplets = triplets[triplets["target_hospital"] == args.target_hospital]
    elif args.level in {5, 6}:
        required = settings.get("lesion_columns" if args.level == 5 else "appearance_columns", [])
        missing = [column for column in required if column not in frame]
        if missing:
            save_table(pd.DataFrame([{"level": args.level, "status": "unavailable", "reason": f"missing validated columns: {missing}"}]), output / "status.csv"); run.complete(status="completed", analysis_status="unavailable"); return
        triplets = pd.DataFrame()
    else:
        triplets = random_pairs(bank, sources, args.target_hospital, 1, seed)[["source_id", "source_split", "source_hospital", "target_hospital", "label", "cross_donor"]]
    save_table(triplets, output / "triplets.csv"); save_table(pd.DataFrame([{"level": args.level, "candidate_sources": len(sources), "accepted_sources": triplets["source_id"].nunique() if not triplets.empty else 0, "coverage": triplets["source_id"].nunique() / max(len(sources), 1)}]), output / "coverage.csv"); run.complete(status="completed", accepted_rows=len(triplets))


if __name__ == "__main__": main()
