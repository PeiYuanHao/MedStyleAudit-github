# Scientific decisions that must remain locked/TBD

The paper intentionally leaves empirical outcomes and development-selected
thresholds unresolved. Before the first full run, select using training/OOD-val
only and record in YAML:

- tissue-mask thresholds and stability acceptance rule;
- lesion-alignment agreement threshold and coordinate/orientation checks;
- K, matching distance/balance thresholds, per-feature calipers, and reuse caps;
- feather width, perceptual seam metric, and blinded QA protocol;
- optimizer schedule, epochs, model-selection rule, and logit IQR floor;
- bootstrap draws, interval type, sparse-cluster fallback, and test families.

Do not fill any result table until its source CSV has been generated. A missing
or failed lesion alignment must remain `unavailable`, not be converted to an
appearance-only substitute.

## Explicit implementation boundaries

- `implemented`: primary integrity/descriptors/matching/coverage,
  counterfactual construction, ERM training, HCS/HCE, robustness, and
  seed/backbone aggregation.
- `partial`: lesion mapping (external WSI/XML required), context-randomized
  quota preparation, planted-cue generation, and ladder levels 5/6.
- `not_implemented`: context-randomized model training/audit, planted-shortcut
  rho calibration, ladder 5/6 matching policies, mitigation, GroupDRO, HED, and
  the remaining P2 suite.

Do not replace `unavailable` or `not_implemented` with an empty result table or
a completed run status.
