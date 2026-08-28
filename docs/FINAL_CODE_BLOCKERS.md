# Final Code Blockers

Items below are intentionally unresolved pending independent review. They do not
block implementation of the reduced suite, but they are recorded so that no
scientific value is silently invented.

## 1. Final donor-slide reuse cap

- **Status:** unresolved.
- **Context:** `configs/matching/primary.yaml` and `configs/final/FINAL_PROTOCOL.yaml`
  carry `donor_slide_reuse_cap: 200000`. This is the pre-existing P0 development
  value and is deliberately nonbinding at the final suite scale (the value exceeds
  any plausible reuse, so it does not constrain matching).
- **Reason not resolved here:** the final *justified* scientific cap must not be
  chosen from model outcomes or HCS/HCE results, and no repository evidence yet
  contains a final, decision-ready value.
- **Required action:** a Codex review must either (a) ratify the existing value as
  the final cap, or (b) supply a justified final value. Until then the value
  remains `200000` (nonbinding) and is flagged here rather than replaced with a
  fabricated number.

## 2. Lesion-aware analysis inputs

- **Status:** conditional/unavailable by default.
- **Context:** lesion-aware analysis is optional and only runs when exact WSI/XML
  mapping is already available and validated. No WSI download or annotation
  retrieval is performed automatically. If no validated mapping exists, the
  pipeline records `status = unavailable` and continues normally.
