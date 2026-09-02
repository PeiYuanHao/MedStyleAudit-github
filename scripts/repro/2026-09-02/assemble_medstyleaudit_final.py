from pathlib import Path
import json
import subprocess
import pandas as pd

ROOT = Path("/root/autodl-tmp/medstyleaudit-experiments")
FINAL = ROOT / "aggregate" / "final"
FINAL.mkdir(parents=True, exist_ok=True)

POPULATIONS = ["hospital1", "hospital2"]

# ============================================================
# Preflight
# ============================================================

for population in POPULATIONS:
    d = ROOT / "aggregate" / population

    required = [
        "run_completeness.csv",
        "global_hcs.csv",
        "global_hce.csv",
        "global_intervals.csv",
        "directed_intervals.csv",
        "INDIVIDUAL_SEEDS.csv",
        "robustness_by_seed.csv",
    ]

    missing = [name for name in required if not (d / name).is_file()]

    if missing:
        raise RuntimeError(
            f"{population} aggregate missing: {missing}"
        )

    completeness = pd.read_csv(
        d / "run_completeness.csv"
    )

    if not (completeness["status"] == "complete").all():
        raise RuntimeError(
            f"{population} aggregate is incomplete"
        )

    print(
        f"{population}: "
        f"{int(completeness['completed_runs'].sum())}/"
        f"{int(completeness['expected_runs'].sum())} PASS"
    )


h2_provenance_path = (
    ROOT
    / "aggregate"
    / "hospital2"
    / "H2_EXPLORATORY_PROVENANCE.json"
)

if not h2_provenance_path.is_file():
    raise RuntimeError(
        "H2 exploratory provenance is missing"
    )

h2_provenance = json.loads(
    h2_provenance_path.read_text(
        encoding="utf-8"
    )
)

assert (
    h2_provenance[
        "confirmatory_h2_matching_status"
    ]
    == "FAIL"
)

print("H2 provenance: PASS")


# ============================================================
# Official-style final assembly
# ============================================================

def read_all(name):
    frames = []

    for population in POPULATIONS:
        path = (
            ROOT
            / "aggregate"
            / population
            / f"{name}.csv"
        )

        if path.is_file():
            frame = pd.read_csv(path)

            if "population" not in frame.columns:
                frame["population"] = population

            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


names = [
    "global_hcs",
    "global_hce",
    "global_intervals",
    "directed_intervals",
    "INDIVIDUAL_SEEDS",
    "run_completeness",
    "robustness_by_seed",
    "matching_results",
]

for name in names:
    frame = read_all(name)

    if not frame.empty:
        frame.to_csv(
            FINAL / f"{name}.csv",
            index=False,
        )


# MAIN RESULTS
hcs = read_all("global_hcs")
hce = read_all("global_hce")

if hcs.empty or hce.empty:
    raise RuntimeError(
        "global HCS/HCE tables are missing"
    )

main = pd.concat(
    [
        hcs.assign(summary_family="hcs"),
        hce.assign(summary_family="hce"),
    ],
    ignore_index=True,
    sort=False,
)

main.to_csv(
    FINAL / "MAIN_RESULTS.csv",
    index=False,
)


# Official aliases
robustness = read_all("robustness_by_seed")
if not robustness.empty:
    robustness.to_csv(
        FINAL / "ROBUSTNESS_RESULTS.csv",
        index=False,
    )

matching = read_all("matching_results")
if not matching.empty:
    matching.to_csv(
        FINAL / "MATCHING_RESULTS.csv",
        index=False,
    )

individual = read_all("INDIVIDUAL_SEEDS")
if not individual.empty:
    individual.to_csv(
        FINAL / "INDIVIDUAL_SEEDS.csv",
        index=False,
    )

global_intervals = read_all("global_intervals")
if not global_intervals.empty:
    global_intervals.to_csv(
        FINAL / "GLOBAL_INTERVALS.csv",
        index=False,
    )

directed_intervals = read_all("directed_intervals")
if not directed_intervals.empty:
    directed_intervals.to_csv(
        FINAL / "DIRECTED_INTERVALS.csv",
        index=False,
    )


# ============================================================
# Explicit analysis provenance
# ============================================================

analysis_provenance = {
    "hospital1": {
        "analysis_class": "confirmatory",
        "matching_status": "PASS",
    },

    "hospital2": {
        "analysis_class": "post_hoc_exploratory",
        "confirmatory_matching_status": "FAIL",
        "matching_regime":
            h2_provenance["matching_regime"],
        "matching_candidate_sources":
            h2_provenance[
                "matching_candidate_sources"
            ],
        "matching_common_support_sources":
            h2_provenance[
                "matching_common_support_sources"
            ],
        "audit_subset_sources":
            h2_provenance[
                "audit_subset_sources"
            ],
        "audit_sampling_seed":
            h2_provenance[
                "audit_sampling_seed"
            ],
    },

    "interpretation_rule": (
        "Hospital-2 results must not be described "
        "as passing the frozen confirmatory matching gate. "
        "They are post-hoc exploratory results on the fixed "
        "10,000-source common-support audit subset."
    ),
}

(
    FINAL
    / "FINAL_ANALYSIS_PROVENANCE.json"
).write_text(
    json.dumps(
        analysis_provenance,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# Copy detailed H2 provenance into final folder
(
    FINAL
    / "H2_EXPLORATORY_PROVENANCE.json"
).write_text(
    json.dumps(
        h2_provenance,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ============================================================
# Manifest
# ============================================================

protocol_hash_path = (
    ROOT / "protocol" / "protocol_sha256.txt"
)

protocol_hash = (
    protocol_hash_path.read_text(
        encoding="utf-8"
    ).strip()
    if protocol_hash_path.is_file()
    else None
)

commit = subprocess.check_output(
    [
        "git",
        "-C",
        "/root/MedStyleAudit-github",
        "rev-parse",
        "HEAD",
    ],
    text=True,
).strip()

manifest = {
    "status": "ready_for_upload",
    "protocol_hash": protocol_hash,
    "git_commit": commit,
    "hospital1_analysis_class":
        "confirmatory",
    "hospital2_analysis_class":
        "post_hoc_exploratory",
    "hospital2_confirmatory_matching_status":
        "FAIL",
    "artifact_manifest": "MANIFEST.json",
}

(
    FINAL / "experiment_manifest.json"
).write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ============================================================
# Final validation
# ============================================================

complete = pd.read_csv(
    FINAL / "run_completeness.csv"
)

assert (
    complete["status"] == "complete"
).all()

assert int(
    complete["completed_runs"].sum()
) == 30

assert set(
    complete["population"].astype(str)
) == {
    "hospital1",
    "hospital2",
}

assert (
    FINAL / "MAIN_RESULTS.csv"
).is_file()

assert (
    FINAL / "FINAL_ANALYSIS_PROVENANCE.json"
).is_file()


print()
print("=" * 72)
print("FINAL ASSEMBLY: PASS")
print("H1 + H2 completed audits = 30/30")
print("Hospital 1 = CONFIRMATORY")
print("Hospital 2 = POST-HOC EXPLORATORY")
print("H2 frozen matching gate = FAIL")
print("output =", FINAL)
print("=" * 72)
