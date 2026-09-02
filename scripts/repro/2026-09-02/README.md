# MedStyleAudit 2026-09-02 Runtime Reproduction Archive

This directory archives the runtime helpers and external wrappers that were
actually used to complete the MedStyleAudit Hospital-2 experiment and final
artifact assembly on AutoDL.

## Frozen experiment provenance

The scientific experiment was executed from the frozen repository revision:

- Git commit: `feede087cae4e0f2ae896cfa7c0cffcdc9030608`
- Frozen protocol SHA256:
  `ab44aaa68bdd7a9e871f3edb1e9a6b959a61068c2b18786588d37e44da2d2b04`

The files in this directory were added to Git only after the experiment had
finished. Their later Git commit must therefore not be confused with the frozen
scientific execution commit above.

## Hospital-2 status

The frozen confirmatory Hospital-2 matching gate FAILED.

The final Hospital-2 analyses are therefore explicitly classified as:

`POST-HOC EXPLORATORY`

The exploratory matching configuration used a compactness caliper of 0.10 and
candidate pool size 128. The resulting full matching did not pass the frozen
90% coverage gate, but provided 69,827 common-support source patches. A fixed
10,000-source audit subset was sampled from this common-support population
before outcome analysis.

The confirmatory matching failure must never be rewritten or reported as a
pass.

## Archived runtime helpers

- `fast_matcher_v2.py`
  Fast NumPy/Numba implementation used for exploratory Hospital-2 matching.

- `benchmark_array_matcher_v2.py`
  Structural/numerical equivalence benchmark used to validate the fast matcher.

- `run_h2_rescue_fast_v2.py`
  Explicit post-hoc exploratory Hospital-2 matching wrapper.

- `run_h2_random_paired_fast.py`
  Faster exact implementation of the Random Paired donor construction.

- `run_predictions_fixed.py`
  Runtime workaround for predictions-only model/device placement.

- `run_h2_remaining_12_parallel.sh`
  Corrected three-seed parallel runner used to complete the remaining
  Hospital-2 audits.

- `run_h2_aggregate_exploratory.py`
  Hospital-2 aggregation wrapper that preserves exploratory matching provenance
  rather than incorrectly reading the confirmatory P0 matching directory.

- `assemble_medstyleaudit_final.py`
  Final Hospital-1 + Hospital-2 result assembly with explicit analysis-class
  provenance.

- `upload_medstyleaudit_final_safe.py`
  Batched, resumable Hugging Face artifact uploader with remote verification.

## Final execution status

- Hospital-1 audits: 15/15 complete
- Hospital-2 audits: 15/15 complete
- Total audits: 30/30 complete
- Hospital-1 analysis class: confirmatory
- Hospital-2 analysis class: post-hoc exploratory
- Final scientific sanity check: PASS
- Hugging Face selected artifacts verified: 675
- Hugging Face final manifest artifacts: 744

No raw Camelyon17/WILDS data, credentials, tokens, or medical source images are
stored in this directory.
