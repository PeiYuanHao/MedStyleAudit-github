from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from medstyleaudit.matching.candidate_bank import CandidateBank
from medstyleaudit.matching.ladder import random_pairs
from medstyleaudit.utils.cli import enforce_final_test_guard
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_table


CONFIG = "configs/matching/primary.yaml"
SUBSET = Path(
    "/root/autodl-tmp/medstyleaudit-experiments/"
    "subsets/hospital2.parquet"
)
OUTPUT = Path(
    "/root/autodl-tmp/medstyleaudit-experiments/"
    "audits/hospital2/random_paired/triplets.parquet"
)

SPLIT = "test"
SEED = 42
VALIDATE_N = 30


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

enforce_final_test_guard(SPLIT, True)

config = load_config(CONFIG)
settings = config["matching"]

k = int(settings["donors_per_source"])

frame = (
    read_table(config["data"]["descriptors"])
    .copy()
    .reset_index(drop=True)
)

subset = read_table(SUBSET)
subset_ids = set(subset["source_id"].tolist())

sources = frame[
    (frame["split"] == SPLIT)
    & frame["source_id"].isin(subset_ids)
].copy()

targets = [
    int(x)
    for x in settings["target_hospitals"][SPLIT]
]

assert sources["source_id"].nunique() == 10000
assert len(sources) == 10000

print("sources =", len(sources), flush=True)
print("targets =", targets, flush=True)
print("K       =", k, flush=True)


# ------------------------------------------------------------
# Build only the donor groups actually needed
# ------------------------------------------------------------

all_groups = frame.groupby(
    ["hospital_id", "label", "split"],
    sort=False,
).indices

needed_keys = set()

for label in sources["label"].unique():

    label = int(label)

    # Within H2 donors come from test.
    for hospital in sources["hospital_id"].unique():
        needed_keys.add(
            (int(hospital), label, SPLIT)
        )

    # Cross donors come from train.
    for target in targets:
        needed_keys.add(
            (target, label, "train")
        )


groups = {}

print("building indexed donor groups...", flush=True)

for key in sorted(needed_keys):

    raw_positions = all_groups.get(key, [])

    positions = np.sort(
        np.asarray(
            raw_positions,
            dtype=np.int64,
        )
    )

    part = frame.iloc[positions]

    source_ids = part["source_id"].to_numpy()
    physical_ids = (
        part["physical_id"]
        .astype(str)
        .to_numpy()
    )

    phys_map = defaultdict(list)
    sid_map = defaultdict(list)

    for local, value in enumerate(physical_ids):
        phys_map[value].append(local)

    for local, value in enumerate(source_ids):
        sid_map[value].append(local)

    groups[key] = {
        "positions": positions,
        "source_ids": source_ids,
        "physical_ids": physical_ids,
        "phys_map": phys_map,
        "sid_map": sid_map,
    }

    print(
        f"group {key}: {len(positions)} donors",
        flush=True,
    )


# ------------------------------------------------------------
# Convert a rank in the filtered candidate list back into
# the original donor-group position.
#
# This preserves CandidateBank.candidates() ordering exactly.
# ------------------------------------------------------------

def excluded_positions(group, physical_id, source_id=None):

    excluded = list(
        group["phys_map"].get(
            str(physical_id),
            (),
        )
    )

    if source_id is not None:
        excluded.extend(
            group["sid_map"].get(
                source_id,
                (),
            )
        )

    if not excluded:
        return np.empty(
            0,
            dtype=np.int64,
        )

    return np.asarray(
        sorted(set(excluded)),
        dtype=np.int64,
    )


def filtered_rank_to_local(ranks, excluded):

    result = np.asarray(
        ranks,
        dtype=np.int64,
    ).copy()

    # Exclusions are sorted.
    # Each removed item shifts all following filtered ranks by 1.
    for value in excluded:
        result += (
            result >= value
        ).astype(np.int64)

    return result


# ------------------------------------------------------------
# Fast implementation preserving official RNG call order:
#
# for each target:
#     rng = default_rng(seed)
#     for each source:
#         rng.choice(within)
#         rng.choice(cross)
# ------------------------------------------------------------

def fast_random_pairs(source_frame, target, show_progress=True):

    rng = np.random.default_rng(SEED)

    rows = []

    iterator = source_frame.itertuples(
        index=False
    )

    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(source_frame),
            desc=f"Random Paired 2->{target}",
            unit="source",
        )

    for source in iterator:

        source_id = source.source_id
        source_hospital = int(source.hospital_id)
        label = int(source.label)
        source_split = str(source.split)
        source_physical = str(source.physical_id)

        donor_split = (
            "train"
            if source_split in {"val", "test"}
            else source_split
        )

        within_key = (
            source_hospital,
            label,
            source_split,
        )

        cross_key = (
            int(target),
            label,
            donor_split,
        )

        within_group = groups[within_key]
        cross_group = groups[cross_key]

        within_excluded = excluded_positions(
            within_group,
            source_physical,
            source_id,
        )

        cross_excluded = excluded_positions(
            cross_group,
            source_physical,
            None,
        )

        within_n = (
            len(within_group["positions"])
            - len(within_excluded)
        )

        cross_n = (
            len(cross_group["positions"])
            - len(cross_excluded)
        )

        if within_n < k or cross_n < k:
            continue

        # EXACT same RNG calls/order as official random_pairs().
        within_ranks = rng.choice(
            within_n,
            k,
            replace=False,
        )

        cross_ranks = rng.choice(
            cross_n,
            k,
            replace=False,
        )

        within_local = filtered_rank_to_local(
            within_ranks,
            within_excluded,
        )

        cross_local = filtered_rank_to_local(
            cross_ranks,
            cross_excluded,
        )

        within_ids = within_group[
            "source_ids"
        ][within_local]

        cross_ids = cross_group[
            "source_ids"
        ][cross_local]

        for donor_index, (
            within_id,
            cross_id,
        ) in enumerate(
            zip(
                within_ids,
                cross_ids,
            )
        ):

            rows.append({
                "source_id":
                    source_id,

                "source_split":
                    source_split,

                "source_hospital":
                    source_hospital,

                "target_hospital":
                    int(target),

                "label":
                    label,

                "within_donor":
                    within_id,

                "cross_donor":
                    cross_id,

                "donor_index":
                    donor_index,
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Exact equivalence test against repository implementation
# ------------------------------------------------------------

print()
print("=" * 72)
print("EXACT EQUIVALENCE CHECK")
print("=" * 72)

bank = CandidateBank.build(
    frame,
    settings["descriptor_columns"],
)

official_sources = bank.frame[
    (bank.frame["split"] == SPLIT)
    & bank.frame["source_id"].isin(subset_ids)
].head(VALIDATE_N)

test_target = targets[0]

official = random_pairs(
    bank,
    official_sources,
    test_target,
    k,
    SEED,
)

fast = fast_random_pairs(
    official_sources,
    test_target,
    show_progress=False,
)

columns = [
    "source_id",
    "source_split",
    "source_hospital",
    "target_hospital",
    "label",
    "within_donor",
    "cross_donor",
    "donor_index",
]

pd.testing.assert_frame_equal(
    official[columns].reset_index(drop=True),
    fast[columns].reset_index(drop=True),
    check_dtype=False,
)

print(
    f"OFFICIAL vs FAST ({VALIDATE_N} sources): EXACT PASS",
    flush=True,
)


# ------------------------------------------------------------
# Full 10k × 3 targets
# ------------------------------------------------------------

print()
print("=" * 72)
print("FULL H2 RANDOM PAIRED")
print("=" * 72)

parts = []

for target in targets:

    selected = fast_random_pairs(
        sources,
        target,
        show_progress=True,
    )

    parts.append(selected)


triplets = pd.concat(
    parts,
    ignore_index=True,
)


# ------------------------------------------------------------
# Same final metadata as scripts/03_random_paired.py
# ------------------------------------------------------------

expected_sources = set(
    sources["source_id"]
)

counts = triplets.groupby(
    ["source_id", "target_hospital"]
).size()

expected_keys = {
    (source_id, target)
    for source_id in expected_sources
    for target in targets
}

observed_keys = set(
    counts.index.tolist()
)

incomplete = sorted(
    key
    for key in expected_keys
    if (
        key not in observed_keys
        or int(counts.loc[key]) != k
    )
)

unexpected = sorted(
    observed_keys - expected_keys
)

if incomplete or unexpected:
    raise RuntimeError(
        "Random Paired incomplete: "
        f"incomplete={incomplete[:20]}, "
        f"unexpected={unexpected[:20]}"
    )


triplets = triplets.reset_index(
    drop=True
)

triplets.insert(
    0,
    "triplet_id",
    [
        f"rp-{SPLIT}-t{i:09d}"
        for i in range(len(triplets))
    ],
)


lookup = (
    bank.frame
    .drop_duplicates("source_id")
    .set_index("source_id")
)

for prefix, id_column in (
    ("source", "source_id"),
    ("within", "within_donor"),
    ("cross", "cross_donor"),
):

    triplets[
        f"{prefix}_slide"
    ] = triplets[id_column].map(
        lookup["slide_id"]
    )

    triplets[
        f"{prefix}_physical_id"
    ] = triplets[id_column].map(
        lookup["physical_id"]
    )


triplets["common_support"] = True


# ------------------------------------------------------------
# Final safety validation
# ------------------------------------------------------------

assert (
    triplets["source_id"].nunique()
    == 10000
)

assert len(triplets) == 90000

assert (
    triplets
    .groupby("source_id")[
        "target_hospital"
    ]
    .nunique()
    .eq(3)
    .all()
)

assert (
    triplets
    .groupby(
        [
            "source_id",
            "target_hospital",
        ]
    )
    .size()
    .eq(3)
    .all()
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

save_table(
    triplets,
    OUTPUT,
)


print()
print("=" * 72)
print("H2 RANDOM PAIRED FAST: PASS")
print("sources =", triplets["source_id"].nunique())
print("rows    =", len(triplets))
print("targets =", targets)
print("output  =", OUTPUT)
print("=" * 72)
