"""Build Random Paired same-label within/cross triplets from the fixed subset.

Random Paired is the only required matching-policy ablation against Balanced
Matched-Triplets. It enforces basic eligibility (same source label, correct
within/cross hospital, physical identity exclusion, paired construction) but does
not optimize morphology matching or matching-distance balance. Selection is
deterministic given the seed and reuses the fixed audit subset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from medstyleaudit.matching.candidate_bank import CandidateBank
from medstyleaudit.matching.ladder import random_pairs
from medstyleaudit.utils.cli import common_parser, enforce_final_test_guard
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_table


def main() -> None:
    parser = common_parser("Build Random Paired matching-policy triplets", "configs/matching/primary.yaml")
    parser.add_argument("--subset", type=Path, default=None, help="Fixed audit subset parquet (source IDs)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    enforce_final_test_guard(args.split, args.allow_final_test)
    config = load_config(args.config)
    settings = config["matching"]
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    frame = read_table(config["data"]["descriptors"])
    bank = CandidateBank.build(frame, settings["descriptor_columns"])
    sources = bank.frame[bank.frame["split"] == args.split].copy()
    if args.subset:
        subset = read_table(args.subset)
        sources = sources[sources["source_id"].isin(set(subset["source_id"]))]
    targets = [int(value) for value in settings["target_hospitals"].get(args.split, [])]
    parts = []
    for target in targets:
        selected = random_pairs(bank, sources, target, int(settings["donors_per_source"]), seed)
        if not selected.empty:
            parts.append(selected)
    triplets = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not triplets.empty:
        lookup = bank.frame.drop_duplicates("source_id").set_index("source_id")
        triplets = triplets.reset_index(drop=True)
        triplets["triplet_id"] = [f"rp-{args.split}-t{index:09d}" for index in range(len(triplets))]
        for prefix, id_column in (("source", "source_id"), ("within", "within_donor"), ("cross", "cross_donor")):
            triplets[f"{prefix}_slide"] = triplets[id_column].map(lookup["slide_id"])
            triplets[f"{prefix}_physical_id"] = triplets[id_column].map(lookup["physical_id"])
        triplets["common_support"] = True
    save_table(triplets, args.output)
    print(f"Random Paired triplets: {len(triplets)} rows from {sources['source_id'].nunique()} subset sources")


if __name__ == "__main__":
    main()
