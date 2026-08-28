# Final Paper Experiment Suite (Reduced)

The final manuscript uses a permanently reduced experiment suite. This document
is the canonical, human-readable description of the **required** experiments.
The machine-readable specification is
[`configs/final/FINAL_PROTOCOL.yaml`](../configs/final/FINAL_PROTOCOL.yaml) and the
executable orchestration is
[`scripts/run_final_suite.py`](../scripts/run_final_suite.py).

## Model

| Setting | Value |
|---|---|
| Architecture | ResNet-50 only |
| Initialization | Random (no ImageNet pretrained weights) |
| Seeds | `11`, `42`, `101` |
| Loss | `BCEWithLogitsLoss` |
| Optimizer | AdamW (`lr=3e-4`, `weight_decay=1e-4`) |
| Scheduler | Cosine |
| Epochs / batch | 20 / 128 |
| Model selection | ID validation (`val_auroc`) |

## Data and hospitals

- Training hospitals: `0`, `3`, `4`.
- OOD validation audit source: **Hospital 1** (validation split).
- Final OOD test audit source: **Hospital 2** (test split, locked until explicit unlock).
- Training hospitals are used only for classifier training, ID-validation model
  selection/normalization, and the training-hospital cross-donor bank.

## Required experiments

### P0 — data and matching (no model)

- Data integrity.
- Morphology descriptors.
- Balanced Matched-Triplet matching.
- Directed coverage and common-support coverage.
- Attrition, aggregate balance, per-feature balance.
- Donor reuse and donor-slide reuse diagnostics.

### P1 — training

ResNet-50 for seeds `11`, `42`, `101`.

### P2 — Hospital 1 / OOD validation audit

- Primary Balanced HCS/HCE (r = 0, feathered boundary).
- Random Paired comparison.
- ROI-only implementation falsification.
- Source buffer `r = 8` (same subset, triplets, checkpoint).
- Hard vs primary feathered boundary (same subset, triplets, checkpoint).

### P3 — Hospital 2 / final OOD test audit

The exact same frozen protocol as Hospital 1, executed only after explicit
Hospital-2 unlock.

## Statistics

Each of the three seeds is bootstrapped independently over physical
source/donor identities. Seed-level point estimates are aggregated with a plain
mean and standard deviation; the three individual seed estimates are preserved.
The seed dimension is never bootstrapped.

## Fixed audit subset

After matching eligibility is established, each held-out hospital selects a fixed
subset of at most `10000` matched eligible sources (sampling seed `2026`),
stratified by label and, where practical, slide. The same selected source IDs are
reused for the primary Balanced audit, Random Paired, ROI-only, `r = 8`, and hard
boundary settings.

## Optional

Lesion-aware analysis runs only if exact WSI/XML mapping is already available and
validated; otherwise its status is `unavailable`.

## Removed (not part of the final paper)

DenseNet-121, ten-seed experiments, planted-shortcut calibration, rho grid,
context-randomized training, GroupDRO, HED augmentation, context-consistency
mitigation, the mitigation benchmark, the full six-level Identification Ladder,
the independent-nearest-neighbor ladder experiment, the appearance-matched
experiment, a second dataset, the HCS-OOD correlation experiment, the dense
source-buffer grid, the dense seam grid, and large hyperparameter sweeps are all
excluded from the final runner and final required scope.
