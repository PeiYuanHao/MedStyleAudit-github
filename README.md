# MedStyleAudit

Reproducible experiment code for **Auditing Hospital-Associated Context
Sensitivity with a Pixel-Identical Label-Defining Region**.

For every Camelyon17-WILDS 96x96 source patch, the central label-defining
32x32 ROI is preserved exactly while matched peripheral context is transplanted
from same-hospital and cross-hospital donors. The code computes Hospital Context
Sensitivity (HCS; excess absolute normalized-logit instability) and Hospital
Context Effect (HCE; signed cross-minus-within shift). These are sensitivity
contrasts, not causal effects of hospital, scanner, stain, or shortcut use.

## Implemented experiment chain

```text
metadata integrity and slide mapping
-> lesion-alignment validation
-> fixed tissue masks and content descriptors
-> balanced matched-triplet selection
-> coverage, attrition, and balance reports
-> exact-ROI hard/feathered counterfactuals
-> random-initialized ResNet-50/DenseNet-121 training
-> frozen-model HCS/HCE audit
-> crossed bootstrap, controls, robustness, OOD summaries
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

## Output layout

```text
/workspace/experiments/medstyleaudit/
├── data_integrity/{wilds_summary.csv,slide_mapping.csv,integrity_report.json}
├── lesion_mapping/{alignment_report.json,lesion_features.csv}
├── descriptors/{descriptors.csv,mask_stability.csv}
├── matching/{triplets,coverage,balance}
├── checkpoints/<backbone>/seed_<seed>/
└── primary_audit/<backbone>/seed_<seed>/<split>/
```

No quantitative paper result is bundled or fabricated. Output schemas are
stable CSV/JSON/YAML files so paper tables and figures remain traceable to raw
experiment records.
