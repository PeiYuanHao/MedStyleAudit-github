# Frozen final experiment plan

The executable order is defined by `configs/final/FINAL_SUITE.yaml` and
`scripts/run_final_suite.py`: P0 integrity/matching/preflight; seed-11 smoke;
2×10 primary training; validation audits; required controls; locked-triplet
robustness and ladder levels 1–4; validation aggregation; explicit hospital-2
unlock; final-test matching/predictions/audits; final tables; private HF upload
and checksum verification.

No classifier outcome may tune matching. No missing pair, seed, or backbone is
silently removed. Lesion-aware analysis is conditional; optional secondary
mitigation and appearance experiments do not block execution.
