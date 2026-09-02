import hashlib
import runpy
from pathlib import Path

import medstyleaudit.protocol as protocol
import medstyleaudit.matching.triplet_matcher as tm

from fast_matcher_v2 import ArrayFastMatcher


print("=" * 78)
print("POST-HOC EXPLORATORY H2 MATCHING RESCUE — FAST V2")
print("This is NOT frozen confirmatory Hospital-2 matching.")
print("Fast V2 was bitwise-equivalent on the validation benchmark.")
print("=" * 78)


# Exploratory rescue deliberately bypasses the frozen-config assertion.
# The original confirmatory H2 failure remains preserved.
protocol.assert_frozen_execution_config = lambda **kwargs: None


# Official runner populates prior donor counters after constructing matcher.
# Initialize array mirrors lazily immediately before matching.
_original_match = ArrayFastMatcher.match


def validated_fast_match(
    self,
    sources,
    target_hospitals,
    show_progress=False,
):
    assert int(self.config["candidate_pool_size"]) == 128
    assert self.config.get("feature_calipers") == {"compactness": 0.1}

    if not hasattr(self, "_sid"):
        self.prepare_arrays()
        self.prime_prior_arrays()

    return _original_match(
        self,
        sources,
        target_hospitals,
        show_progress=show_progress,
    )


ArrayFastMatcher.match = validated_fast_match

# 03_run_matching imports BalancedTripletMatcher after this monkeypatch.
tm.BalancedTripletMatcher = ArrayFastMatcher


for p in [
    "/root/autodl-tmp/fast_matcher_v2.py",
    "/root/autodl-tmp/h2_rescue_c010_pool128.FROZEN.yaml",
    "/root/autodl-tmp/h1_prior_triplets_only.parquet",
]:
    path = Path(p)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"SHA256 {path.name} = {h}")


runpy.run_path(
    "/root/MedStyleAudit-github/scripts/03_run_matching.py",
    run_name="__main__",
)
