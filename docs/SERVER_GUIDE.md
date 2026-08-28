# Final AutoDL execution guide

The checkout is code-only. Raw Camelyon17 data and working artifacts stay outside
it. The exact private Hugging Face Dataset namespace is
`PeiyuanHao/MedStyleAudit-Experiments`; the GitHub namespace is different.

```bash
git clone https://github.com/PeiYuanHao/MedStyleAudit-github.git
cd MedStyleAudit-github
export MEDSTYLE_DATA_ROOT="/root/autodl-tmp/datasets"
export MEDSTYLE_OUTPUT_ROOT="/root/autodl-tmp/medstyleaudit-experiments"
export MEDSTYLE_HF_REPO="PeiyuanHao/MedStyleAudit-Experiments"
test -n "${HF_TOKEN:?HF_TOKEN must already be available in the environment}"
bash scripts/autodl_setup.sh
python scripts/hf_create_repository.py
bash scripts/autodl_run_final_suite.sh
```

If the instance has multiple GPUs, list them explicitly; the runner assigns at
most one subprocess to each device and parallelizes every per-model phase:

```bash
bash scripts/autodl_run_final_suite.sh --devices cuda:0 cuda:1 cuda:2 cuda:3
```

On a single RTX 2080 Ti, use the default command. Training uses AMP and optimized
DataLoader settings automatically. Completed stages and completed planted-cue
rho values are resumed without recomputation.

The first run ends after validation aggregation with `final_test=locked`. Resume
only after choosing to open the final test:

```bash
bash scripts/autodl_run_final_suite.sh --allow-final-test
```

If remote checksum verification fails, AutoDL remains running. Inspect
`logs/final_suite/`, then retry:

```bash
python scripts/hf_upload_artifacts.py --root "${MEDSTYLE_OUTPUT_ROOT}"
python scripts/hf_verify_artifacts.py
```

The wrapper never deletes local results after upload.
