# Frozen final experiment plan

The executable order is defined by `configs/final/FINAL_SUITE.yaml` and
`scripts/run_final_suite.py`: P0 integrity/matching/preflight; seed-11 smoke;
2×10 primary training; validation audits; required controls; locked-triplet
robustness and ladder levels 1–4; validation aggregation; explicit hospital-2
unlock; final-test matching/predictions/audits; final tables; private HF upload
and checksum verification.

The trained-control calibration scope is fixed in `FINAL_SUITE.yaml` to
ResNet-50 seed 42. ROI-only remains an inference-only falsification for every
primary model. Thus the required suite trains 20 primary models, one
context-randomized model, and five planted-shortcut models rather than repeating
the six calibration models over every primary backbone/seed combination.

No classifier outcome may tune matching. No missing pair, seed, or backbone is
silently removed. Lesion-aware analysis is conditional; optional secondary
mitigation and appearance experiments do not block execution.
