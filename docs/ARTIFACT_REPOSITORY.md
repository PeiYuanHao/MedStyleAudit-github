# Final experiment artifact repository

The private Hugging Face Dataset repository `PeiyuanHao/MedStyleAudit-Experiments`
is the persistent store for **derived** final-experiment artifacts. GitHub contains
only source code, frozen configs, tests, and documentation. Raw Camelyon17-WILDS
patches, WSI files, XML annotations, and unchanged source shards must remain on
the execution server and must never be uploaded.

The dataset card template is [HF_DATASET_CARD.md](HF_DATASET_CARD.md). Each upload
publishes `MANIFEST.json` last, with byte counts and SHA256 checksums. Repeated
uploads skip files whose manifest checksum is unchanged. Downloads and remote
verification fail on any checksum mismatch.
