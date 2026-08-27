# Server guide

Recommended layout:

```text
/workspace/MedStyleAudit                            # Git repository: code only
/workspace/datasets/huggingface/Camelyon17-WILDS   # persistent dataset disk
/workspace/datasets/camelyon17                     # optional original WSIs
/workspace/experiments/medstyleaudit               # outputs and HF cache
```

Initialize and export the server paths:

```bash
export MEDSTYLE_DATA_ROOT=/workspace/datasets
export MEDSTYLE_OUTPUT_ROOT=/workspace/experiments/medstyleaudit
export HF_HOME=/workspace/experiments/medstyleaudit/cache/huggingface
bash scripts/server_setup.sh
```

Download data directly to the persistent dataset disk. Never download into
`/workspace/MedStyleAudit`:

```bash
python -m pip install -U "huggingface_hub[hf_xet]"
hf download wltjr1007/Camelyon17-WILDS \
  --repo-type dataset \
  --local-dir /workspace/datasets/huggingface/Camelyon17-WILDS
```

Build with `docker compose -f docker/docker-compose.yml build`. Compose now
requires both storage variables and refuses implicit repository-local mounts.
It mounts datasets read-only and experiments read/write. Run P0 on CPU before
renting GPU time.

Use `--dry-run` for one-seed pipeline validation. Training supports `--resume`;
every epoch writes `metrics.csv` and `last.ckpt`, while model-selection updates
`best.ckpt` and saved validation predictions.

Before deleting a rented server, verify that `/workspace/datasets` and
`/workspace/experiments` are attached persistent volumes or copy the complete
experiment directory elsewhere.
