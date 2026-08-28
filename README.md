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

- Backbones: random-initialized ResNet-50 and DenseNet-121.
- Seeds: `11, 23, 42, 57, 71, 89, 101, 131, 173, 211`.
- Matching: three donors/source, `lambda_balance=2.0`, `lambda_pair=0.25`,
  `tau_distance=6.0`, `tau_balance=1.0`, candidate pool 64, donor cap 20.
- Controls: ROI-only, context-randomized, planted shortcut at
  `rho=[0,.25,.50,.75,1]`. ROI-only is evaluated for all primary models; the
  two controls that require new training use the predeclared ResNet-50 seed 42
  calibration scope and are not repeated across all 20 primary runs.
- Robustness: locked-triplet buffers `[0,4,8,16]`, hard versus primary feathered
  seam, and lesion-aware only when alignment is validated.
- Identification ladder: levels 1–4 required; lesion-aware conditional.

The runner executes: environment and tests; integrity and P0 matching; preflight;
one full seed-11 smoke path; both backbones × ten seeds; validation audits;
controls and robustness; validation aggregation; explicit final-test unlock;
hospital-2 matching/predictions/audits; final tables; HF upload and verification.
Every stage has a completion marker and validated expected outputs. Re-running
resumes completed stages; `--force` intentionally recomputes them.

CUDA training uses deterministic FP16 AMP, channels-last tensors, pinned-memory
workers, and persistent prefetching. Counterfactual inference batches multiple
triplets per GPU call. On a multi-GPU server, independent primary runs can be
scheduled with:

```bash
bash scripts/autodl_run_final_suite.sh --devices cuda:0 cuda:1 cuda:2 cuda:3
```

For manual primary-only sharding, run `python scripts/run_final_suite.py` with
`--backbones` and/or `--seeds`. A shard stops after its validation audits and
deliberately does not aggregate, open the final test, or upload. Do not use the
AutoDL wrapper for a shard because the wrapper shuts the instance down after a
successful command.

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
dataset. They do not block the suite. Unavailable lesion alignment records
`lesion_aware_status=unavailable` and does not block the core audit.

See [`docs/ARTIFACT_REPOSITORY.md`](docs/ARTIFACT_REPOSITORY.md) and
[`docs/SERVER_GUIDE.md`](docs/SERVER_GUIDE.md).
