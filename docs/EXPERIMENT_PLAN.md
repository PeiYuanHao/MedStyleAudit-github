# Frozen final experiment plan (reduced)

The executable order is defined by `configs/final/FINAL_SUITE.yaml` and
`scripts/run_final_suite.py`:

1. **P0** — data integrity, descriptors, balanced matched-triplet matching,
   matching diagnostics, fixed Hospital-1 subset, preflight.
2. **P1** — ResNet-50 training for seeds `11`, `42`, `101`.
3. **P2** — Hospital-1 audit matrix per seed: primary Balanced HCS/HCE, Random
   Paired, ROI-only, `r=8` buffer, hard boundary.
4. **P3** — Hospital-1 aggregation.
5. **P4** — explicit Hospital-2 unlock.
6. **P5** — Hospital-2 matching/subset/audits using the frozen protocol.
7. **P6** — final aggregation, Hugging Face upload, manifest generation.

The reduced suite is ResNet-50 only, three seeds, and two held-out audit
populations (Hospital 1 / validation, Hospital 2 / test). Every robustness
setting reuses the same fixed audit subset, the same balanced triplets, and the
same checkpoint; matching is never rematched at `r=8` or the hard boundary.

No classifier outcome may tune matching. No missing pair, seed, or population is
silently removed. Lesion-aware analysis is conditional and optional; deleted
experiments (DenseNet-121, ten-seed runs, planted shortcut, context-randomized
training, the identification ladder, and secondary mitigation/appearance work)
do not appear in the final runner.
