# MedStyleAudit

Final experiment execution code for **Auditing Hospital-Associated Context
Sensitivity with a Pixel-Identical Label-Defining Region**.

The methodology is frozen in
[`configs/final/FINAL_PROTOCOL.yaml`](configs/final/FINAL_PROTOCOL.yaml). For
each 96×96 Camelyon17-WILDS patch, the central 32×32 label-defining ROI is
preserved exactly while matched peripheral context is transplanted. HCS and HCE
are sensitivity contrasts; they do not causally identify a hospital, scanner,
stain, or shortcut mechanism.

## Repository responsibilities

| Location | Responsibility | Never stored there |
|---|---|---|
| GitHub — `PeiYuanHao/MedStyleAudit-github` | Code, configs, tests, protocol specification, lightweight docs | Checkpoints, predictions, descriptors, triplets, final results |
| Private HF Dataset — `PeiyuanHao/MedStyleAudit-Experiments` | Derived experiment artifacts, checksums, checkpoints, predictions, audit records, final tables | Raw WILDS images, source shards, WSIs, XML annotations |
| Server local disk | Raw datasets, working cache, resumable outputs, logs | Nothing is deleted after upload |

```text
GitHub (code / config / tests)
   |
   | git clone
   v
AutoDL Server
   | raw datasets remain local
   | resumable final experiments
   v
Hugging Face Dataset (derived artifacts / checkpoints / results)
```

Every HF upload writes `MANIFEST.json` last. It records paths, SHA256, bytes,
experiment, backbone, seed, split, Git commit, and protocol hash. Repeated
uploads skip checksum-identical files; verification fails on a mismatch.

## Frozen suite

The final manuscript uses a permanently reduced suite. The canonical description
is [`docs/FINAL_PAPER_EXPERIMENTS.md`](docs/FINAL_PAPER_EXPERIMENTS.md); the
machine-readable protocol is
[`configs/final/FINAL_PROTOCOL.yaml`](configs/final/FINAL_PROTOCOL.yaml).

- Backbone: random-initialized ResNet-50 only (no pretrained weights).
- Seeds: `11, 42, 101`.
- Matching: three donors/source, `lambda_balance=2.0`, `lambda_pair=0.25`,
  `tau_distance=6.0`, `tau_balance=1.0`, candidate pool 64, donor cap 20.
- Audit source hospitals: Hospital 1 (OOD validation) and Hospital 2 (final
  OOD test, locked until explicit unlock). Training hospitals 0/3/4 serve only
  as the cross-donor bank and ID-validation normalization.
- Low-cost checks: Random Paired, ROI-only, `r=8` buffer, and hard vs feathered
  boundary — each reusing the same fixed audit subset, triplets, and checkpoint.
- Lesion-aware analysis is optional and only runs when WSI/XML alignment is
  already validated.

The runner executes: environment and tests; integrity and P0 matching; preflight;
the fixed Hospital-1 subset; ResNet-50 training for seeds 11/42/101; the Hospital-1
audit matrix; Hospital-1 aggregation; explicit final-test unlock; Hospital-2
matching/subset/audits; final aggregation; Hugging Face upload and verification.
Every stage has a completion marker and validated expected outputs. Re-running
resumes completed stages; `--force` intentionally recomputes them.

CUDA training uses deterministic FP16 AMP, channels-last tensors, pinned-memory
workers, and persistent prefetching. Counterfactual inference batches multiple
triplets per GPU call. On a multi-GPU server, primary training, the Hospital-1
audit matrix, and the final-test audit matrix are all scheduled across the
listed devices:

```bash
bash scripts/autodl_run_final_suite.sh --devices cuda:0 cuda:1 cuda:2 cuda:3
```

The reduced suite has no manual backbone/seed sharding; the full suite resumes
in place via `--resume` and can be recomputed with `--force`.

## Installation and tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

Unit tests do not require Camelyon17. NumPy is constrained below 2.0 to avoid
binary ABI mismatches with scientific Python wheels.

## AutoDL execution

```bash
export MEDSTYLE_DATA_ROOT="/root/autodl-tmp/datasets"
export MEDSTYLE_OUTPUT_ROOT="/root/autodl-tmp/medstyleaudit-experiments"
export MEDSTYLE_HF_REPO="PeiyuanHao/MedStyleAudit-Experiments"
test -n "${HF_TOKEN:?HF_TOKEN must already be available in the environment}"
export MEDSTYLE_OPERATOR="operator-name"
export MEDSTYLE_SHUTDOWN_ON_FAILURE="1"
```

`HF_TOKEN` is read only from the environment and is never stored in source or
configuration. `MEDSTYLE_OPERATOR` is recommended; `MEDSTYLE_CONDA_ENV` and
`MEDSTYLE_SHUTDOWN_ON_FAILURE` are optional.

Create/confirm the private dataset repository:

```bash
export MEDSTYLE_HF_REPO="PeiyuanHao/MedStyleAudit-Experiments"
python scripts/hf_create_repository.py
```

Run through validation while hospital 2 remains locked:

```bash
bash scripts/autodl_run_final_suite.sh
```

Then explicitly resume and open the final test:

```bash
bash scripts/autodl_run_final_suite.sh --allow-final-test
```

The wrapper pulls with `--ff-only`, requires a clean commit, preserves local
artifacts, and verifies the HF copy before shutdown. HF verification failure
keeps the instance running.

Optional secondary experiments are appearance-matched ladder level 6, GroupDRO,
HED augmentation, context-consistency mitigation, and a second external medical
dataset. They are legacy/optional and are **not** part of the final paper suite;
they do not appear in the final runner and do not block execution. Unavailable
lesion alignment records `lesion_aware_status=unavailable` and does not block the
core audit.

See [`docs/FINAL_PAPER_EXPERIMENTS.md`](docs/FINAL_PAPER_EXPERIMENTS.md),
[`docs/FINAL_CODE_BLOCKERS.md`](docs/FINAL_CODE_BLOCKERS.md),
[`docs/ARTIFACT_REPOSITORY.md`](docs/ARTIFACT_REPOSITORY.md) and
[`docs/SERVER_GUIDE.md`](docs/SERVER_GUIDE.md).
