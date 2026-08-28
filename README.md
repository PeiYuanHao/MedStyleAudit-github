# MedStyleAudit

Reproducible experiment code for **Auditing Hospital-Associated Context
Sensitivity with a Pixel-Identical Label-Defining Region**.

For every Camelyon17-WILDS 96x96 source patch, the central label-defining
32x32 ROI is preserved exactly while matched peripheral context is transplanted
from same-hospital and cross-hospital donors. The code computes Hospital Context
Sensitivity (HCS; excess absolute normalized-logit instability) and Hospital
Context Effect (HCE; signed cross-minus-within shift). These are sensitivity
contrasts, not causal effects of hospital, scanner, stain, or shortcut use.

## Implementation status

Implemented end-to-end:

- metadata integrity, descriptors, balanced matching, coverage, and balance;
- exact-ROI counterfactual construction and ERM training;
- primary HCS/HCE, pair completeness checks, robustness, and multi-run aggregation.

Partially implemented:

- lesion mapping (the code path is connected, but results are `unavailable` without exact WSI/XML assets and validated coordinates);
- context-randomized control (quota preparation only; training/audit are `not_implemented`);
- planted shortcut (cue utilities only; rho training/audit/calibration are `not_implemented`);
- identification ladder levels 5/6 (`unavailable` when inputs are absent, otherwise `not_implemented`).

Secondary and not implemented: context-consistency mitigation, GroupDRO, HED
augmentation, and the complete P2 suite. Placeholder commands report
`not_implemented`; they do not report successful experiments.

## Implemented primary chain

```text
metadata integrity and slide mapping
-> lesion-alignment validation
-> fixed tissue masks and content descriptors
-> balanced matched-triplet selection
-> coverage, attrition, and balance reports
-> exact-ROI hard/feathered counterfactuals
-> random-initialized ResNet-50/DenseNet-121 training
-> frozen-model HCS/HCE audit
-> crossed bootstrap, primary robustness, OOD summaries
-> configured seed/backbone aggregation with a missing-run ledger
```

Every executable run creates `run_info.json`, `config_resolved.yaml`, `run.log`,
and task-specific CSV/JSON records. Completed runs are not overwritten unless
`--overwrite` is passed. Hospital 2 (`test`) requires `--allow-final-test`.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

The repository contains code only. On the server, data, caches, checkpoints,
logs, and result tables must use persistent directories outside the checkout:

```bash
export MEDSTYLE_DATA_ROOT=/workspace/datasets
export MEDSTYLE_OUTPUT_ROOT=/workspace/experiments/medstyleaudit
export HF_HOME=/workspace/experiments/medstyleaudit/cache/huggingface
bash scripts/server_setup.sh
```

## Large datasets (manual download only)

The loaders never initiate a large download. Download directly on the server,
not on a development machine and not inside the repository.

- Camelyon17-WILDS Parquet mirror: download to
  `/workspace/datasets/huggingface/Camelyon17-WILDS/`.
- Original CAMELYON17 WSIs/annotations: <https://camelyon17.grand-challenge.org/Data/>;
  these are optional for the core audit and should be deferred.

See [docs/DATASETS.md](docs/DATASETS.md). The WILDS labeled archive is roughly
10 GB compressed/15 GB on disk; original WSIs are substantially larger. Neither
is downloaded or tracked by this repository.

The current CodaLab bundle is unreliable, so the server-first examples use the
complete Hugging Face Parquet mirror and `*_hf.yaml` configs.

## Staged execution

Run the CPU feasibility gate first:

```bash
python scripts/00_data_integrity.py --config configs/data/camelyon17_hf.yaml
python scripts/01_validate_lesion_mapping.py --config configs/data/camelyon17_hf.yaml
python scripts/02_build_descriptors.py --config configs/data/camelyon17_hf.yaml
python scripts/03_run_matching.py --config configs/matching/primary.yaml
python scripts/04_check_matching_coverage.py --config configs/matching/primary.yaml
```

Review `${MEDSTYLE_OUTPUT_ROOT}/matching/coverage_review/feasibility_gate.json` before GPU
training. Then run a one-seed smoke test:

Phase 3 first writes `matching/coverage/reuse_capacity.csv` and refuses to enter
the source loop when the prespecified donor or donor-slide reuse caps have
insufficient metadata-only capacity. The operational slide cap is deliberately
nonbinding relative to patch-level reuse for the P0 feasibility run; its final
value remains a protocol parameter to lock before model-outcome access.

```bash
python scripts/05_train_erm.py --config configs/models/resnet50_hf.yaml --seed 42 --device cuda --dry-run
python scripts/06_run_primary_audit.py --config configs/audit/primary.yaml \
  --model-config configs/models/resnet50_hf.yaml \
  --checkpoint ${MEDSTYLE_OUTPUT_ROOT}/checkpoints/resnet50/seed_0042/best.ckpt \
  --id-logits ${MEDSTYLE_OUTPUT_ROOT}/checkpoints/resnet50/seed_0042/id_validation_predictions.csv \
  --device cuda --dry-run
```

Only after the full path passes should all ten configured seeds, the second
backbone, controls, and robustness grids be launched. Lesion-aware results are
reported unavailable unless annotation alignment passes the configured check.

After the configured runs finish, aggregate them without silently dropping a
seed or backbone:

```bash
python scripts/12_aggregate_experiments.py --split val
```

## AutoDL 2080 Ti workflow

The supplied AutoDL setup preserves the base image's PyTorch/CUDA installation
and uses `/root/autodl-tmp` for datasets and experiment outputs:

```bash
git clone git@github.com:PeiYuanHao/MedStyleAudit-github.git
cd MedStyleAudit-github
bash scripts/autodl_setup.sh
```

Configure non-interactive GitHub authentication (SSH key or a credential helper)
before starting an unattended job. The wrapper runs the command, exports only
GitHub-safe CSV/JSON/YAML/log records to `results/<tag>`, and attempts to commit
and push them to `main`. Its exit trap invokes `/usr/bin/shutdown` whenever the
job ends, whether the experiment or Git upload succeeds or fails:

```bash
MEDSTYLE_RESULT_TAG=resnet50-seed42-val \
bash scripts/autodl_run_export_shutdown.sh \
  python scripts/06_run_primary_audit.py --config configs/audit/primary.yaml \
  --model-config configs/models/resnet50_hf.yaml \
  --checkpoint /root/autodl-tmp/medstyleaudit-experiments/checkpoints/resnet50/seed_0042/best.ckpt \
  --id-logits /root/autodl-tmp/medstyleaudit-experiments/checkpoints/resnet50/seed_0042/id_validation_predictions.csv \
  --device cuda
```

Only records created or modified by the wrapped job are considered. Datasets,
caches, checkpoints, model weights, files over 20 MiB, and exports beyond
100 MiB total are not committed; `export_manifest.json` records every copied or skipped file. Set
`MEDSTYLE_SKIP_SHUTDOWN=1` for a safe wrapper rehearsal. AutoDL documents
[`/usr/bin/shutdown` as its post-job shutdown command](https://www.autodl.com/docs/save_money/).

## Output layout

```text
/workspace/experiments/medstyleaudit/
├── data_integrity/{wilds_summary.csv,slide_mapping.csv,patch_mapping.csv,integrity_report.json}
├── lesion_mapping/{alignment_report.json,lesion_features.csv}
├── descriptors/{descriptors.csv,mask_stability.csv}
├── matching/{triplets,coverage,balance}
├── checkpoints/<backbone>/seed_<seed>/
├── primary_audit/<backbone>/seed_<seed>/<split>/
└── aggregate/<split>/{combined_*.csv,seed_stability.csv,missing_runs.csv}
```

No quantitative paper result is bundled or fabricated. Output schemas are
stable CSV/JSON/YAML files so paper tables and figures remain traceable to raw
experiment records.
