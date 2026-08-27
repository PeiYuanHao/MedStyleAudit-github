# Required datasets

This project never downloads large medical datasets automatically.

## Camelyon17-WILDS

- Direct/manual download page: https://wilds.stanford.edu/downloads/
- Dataset description: https://wilds.stanford.edu/datasets/#camelyon17
- WILDS documentation: https://wilds.stanford.edu/get_started/
- Place under: `${MEDSTYLE_DATA_ROOT}/wilds/camelyon17_v1.0/`

The released WILDS patch dataset is sufficient for the primary classifier,
mask descriptor, matching, and counterfactual experiments.

The official WILDS repository reports about 10 GB compressed and 15 GB on disk
for labeled Camelyon17. Do not add `--download` to project commands: the project
intentionally fails if the expected folder is absent.

### Working fallback when the official CodaLab bundle is unavailable

The official CodaLab REST bundle was returning HTTP 500 when checked on
2026-08-27. A community Hugging Face mirror contains 455,954 examples in 10.7
GB of Parquet shards:

- https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS
- Download to: `${MEDSTYLE_DATA_ROOT}/huggingface/Camelyon17-WILDS/`

Install the downloader on the server and download with resume support:

```bash
python -m pip install -U "huggingface_hub[hf_xet]"
hf download wltjr1007/Camelyon17-WILDS \
  --repo-type dataset \
  --local-dir /workspace/datasets/huggingface/Camelyon17-WILDS
```

Then use `configs/data/camelyon17_hf.yaml` and the `*_hf.yaml` model configs.
This is a community mirror, so the integrity stage still verifies sample,
hospital, slide, and split counts before experiments.

## Original CAMELYON17 whole-slide images and annotations

- Challenge/data page: https://camelyon17.grand-challenge.org/Data/
- Place WSIs under: `${MEDSTYLE_DATA_ROOT}/camelyon17/images/`
- Place XML annotations under: `${MEDSTYLE_DATA_ROOT}/camelyon17/annotations/`

The original WSIs and annotations are only required for slide mapping and the
lesion-aware experiment. If mapping cannot be validated, the program records
that analysis as unavailable; it never invents lesion features.

The primary matched-context audit can proceed without these original WSIs.
Download them later only if lesion-aware analysis is required.

After placing data, run
`python scripts/00_data_integrity.py --config configs/data/camelyon17_hf.yaml --dry-run`
before full extraction. Dataset archives, patches, WSIs, caches, derived arrays,
checkpoints, and outputs must remain outside Git.
