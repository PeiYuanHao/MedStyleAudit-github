# Data policy

No dataset belongs inside this Git repository. The `data/` directory contains
documentation only.

On the experiment server, use a persistent data volume:

```text
/workspace/datasets/
├── huggingface/Camelyon17-WILDS/
│   └── data/*.parquet
└── camelyon17/
    ├── images/       # optional original WSIs
    └── annotations/  # optional XML annotations
```

Set `MEDSTYLE_DATA_ROOT=/workspace/datasets`. Do not symlink or copy the dataset
into the repository checkout. See `docs/DATASETS.md` for download commands.
