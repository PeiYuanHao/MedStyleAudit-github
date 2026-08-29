# Final Code Blockers

Items below record the disposition of the final code-review blockers so that no
scientific value is silently invented.

## 1. Final donor-slide reuse cap

- **Status:** **RESOLVED — ratified as a nonbinding operational ceiling.**
- **Context:** `configs/matching/primary.yaml` and `configs/final/FINAL_PROTOCOL.yaml`
  retain `donor_slide_reuse_cap: 200000`.
- **Matching-only evidence:** commit `6dfd4f3398eab7bf271324bba91e725e8f0467cc`
  introduced 200000 explicitly as a metadata-feasible operational cap to prevent
  the earlier cap from collapsing matching. The traceable P0 artifacts added by
  commit `16cfd0e` record that exact configuration, a feasible metadata-only reuse
  capacity gate (628272 required versus 6040020 available donor uses globally),
  and a clean completed matching run with 104712 candidate comparisons and 304671
  accepted triplets. No model prediction, AUROC, HCS, or HCE result was consulted.
- **Final policy:** 200000 is intentionally retained only as a fail-safe operational
  ceiling. It is not claimed to provide meaningful slide-level regularization and
  was not selected using classifier-dependent outcomes. The scientific reuse
  constraint remains the separately frozen donor-patch cap; this ceiling must not
  be interpreted as a tuned slide concentration hyperparameter.

## 2. Lesion-aware analysis inputs

- **Status:** conditional/unavailable by default.
- **Context:** lesion-aware analysis is optional and only runs when exact WSI/XML
  mapping is already available and validated. No WSI download or annotation
  retrieval is performed automatically. If no validated mapping exists, the
  pipeline records `status = unavailable` and continues normally.
