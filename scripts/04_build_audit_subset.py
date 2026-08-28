"""Build the fixed, deterministic audit source subset for one held-out hospital.

Runs after balanced matching and before any classifier outcome is read. The same
selected source IDs are reused for every audit setting (primary, random paired,
ROI-only, r=8, hard boundary) so robustness settings are never independently
resampled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from medstyleaudit.audit.subset import matched_eligible_sources, select_audit_subset
from medstyleaudit.utils.cli import common_parser, enforce_final_test_guard
from medstyleaudit.utils.io import read_table, save_table
from medstyleaudit.utils.paths import experiment_path


def main() -> None:
    parser = common_parser("Select the fixed audit source subset", "configs/matching/primary.yaml")
    parser.add_argument("--triplets", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--sampling-seed", type=int, default=2026)
    parser.add_argument("--triplets-out", type=Path, default=None, help="Optionally write the subset-filtered triplet ledger")
    args = parser.parse_args()
    enforce_final_test_guard(args.split, args.allow_final_test)
    triplets = read_table(args.triplets or experiment_path("p0/matching/triplets.parquet"))
    triplets = triplets[triplets["source_split"] == args.split] if "source_split" in triplets else triplets
    eligible = matched_eligible_sources(triplets, args.split)
    subset = select_audit_subset(eligible, sample_size=args.sample_size, seed=args.sampling_seed)
    save_table(subset, args.output)
    if args.triplets_out:
        ids = set(subset["source_id"])
        save_table(triplets[triplets["source_id"].isin(ids)].reset_index(drop=True), args.triplets_out)
    print(f"Audit subset: {len(subset)} selected from {len(eligible)} matched eligible sources")


if __name__ == "__main__":
    main()
