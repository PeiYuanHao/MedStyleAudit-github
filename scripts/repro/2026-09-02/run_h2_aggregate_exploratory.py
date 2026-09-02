from pathlib import Path
import json
import runpy

import medstyleaudit.utils.paths as paths

ROOT = Path("/root/autodl-tmp/medstyleaudit-experiments")

H2_MATCHING = (
    ROOT
    / "exploratory"
    / "h2_c010_pool128_fastv2"
)

AGGREGATE = ROOT / "aggregate" / "hospital2"

original_experiment_path = paths.experiment_path

def redirected_experiment_path(relative=""):
    value = str(relative).replace("\\", "/")

    if value == "p0/matching":
        return H2_MATCHING

    if value.startswith("p0/matching/"):
        tail = value[len("p0/matching/"):]
        return H2_MATCHING / tail

    return original_experiment_path(relative)

paths.experiment_path = redirected_experiment_path

runpy.run_path(
    "/root/MedStyleAudit-github/scripts/12_aggregate_experiments.py",
    run_name="__main__",
)

provenance = {
    "population": "hospital2",
    "analysis_class": "post_hoc_exploratory",
    "confirmatory_h2_matching_status": "FAIL",
    "matching_regime": "h2_c010_pool128_common_support_subset",
    "matching_candidate_sources": 85054,
    "matching_common_support_sources": 69827,
    "matching_common_support_rate": float(69827 / 85054),
    "audit_subset_sources": 10000,
    "audit_sampling_seed": 2026,
    "matching_source": str(H2_MATCHING),
}

AGGREGATE.mkdir(parents=True, exist_ok=True)

(
    AGGREGATE / "H2_EXPLORATORY_PROVENANCE.json"
).write_text(
    json.dumps(provenance, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("=" * 70)
print("H2 AGGREGATION: PASS")
print("confirmatory H2 matching = FAIL")
print("analysis class            = POST-HOC EXPLORATORY")
print("=" * 70)
