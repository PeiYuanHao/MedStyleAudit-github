# Unattended AutoDL Final Run

The final suite runs in two separately authorized phases. Both phases finalize by
backing up available artifacts, exporting small results, recording status, syncing
the filesystem, and attempting an immediate AutoDL shutdown. A failed experiment or
backup does not keep the instance running.

## Preparation

Prepare the dataset and Python environment, ensure the source checkout is clean,
and configure authentication without putting tokens in command lines or files.
`HF_TOKEN` is mandatory: unattended execution refuses to start the expensive final
suite without it, then records the preflight failure and follows the normal GitHub
export, sync, and shutdown path.

```bash
export HF_TOKEN=...
export MEDSTYLE_HF_REPO=PeiyuanHao/MedStyleAudit-Experiments
export MEDSTYLE_DATA_ROOT=/root/autodl-tmp/datasets
export MEDSTYLE_OUTPUT_ROOT=/root/autodl-tmp/medstyleaudit-experiments
```

GitHub export uses the existing authenticated `origin` remote. It is enabled by
default; set `MEDSTYLE_GITHUB_EXPORT=0` only when a GitHub snapshot is intentionally
not wanted. Individual GitHub-export files are limited to 10 MiB by default. The
limit can be lowered with `MEDSTYLE_GITHUB_MAX_FILE_BYTES`.

## Phase 1: Hospital 1

From the repository root, run:

```bash
nohup bash scripts/autodl_run_final_suite.sh \
  --phase hospital1 \
  --devices cuda:0 cuda:1 \
  > /root/autodl-tmp/hospital1_launcher.log 2>&1 &
```

This runs Hospital 1 without `--allow-final-test`, attempts Hugging Face backup and
verification, pushes a small GitHub snapshot branch, and shuts down the instance.
Hospital 2 remains locked. Experiment, HF, or GitHub failures are recorded but do
not prevent the shutdown attempt.

## Manual Hospital-2 unlock

Restart the server, enter the clean repository checkout, and explicitly run:

```bash
python scripts/15_unlock_final_test.py
```

The unlock succeeds only after the current Hospital-1 prerequisites, Git commit,
protocol hash, and preflight pass. The unattended wrapper never runs this command.

## Phase 2: Hospital 2

After the explicit unlock:

```bash
nohup bash scripts/autodl_run_final_suite.sh \
  --phase hospital2 \
  --devices cuda:0 cuda:1 \
  > /root/autodl-tmp/hospital2_launcher.log 2>&1 &
```

The wrapper first validates the existing unlock, then runs the final suite with
`--resume --allow-final-test`. Missing or stale authorization fails before Hospital
2 execution, but final backup and shutdown are still attempted.

On successful Hospital-2 completion, `run_final_suite.py` has already performed its
normal full Hugging Face upload and verification. When the current run's
`.hf_verified` marker is valid, the unattended finalizer reuses that evidence rather
than repeating the full upload/verification. It still uploads the latest
`autodl_final_status.json` separately. Hospital-2 failures always run the recovery
upload and verification, regardless of any old marker.

## Artifacts and recovery

The wrapper log is stored at:

```text
${MEDSTYLE_OUTPUT_ROOT}/logs/final_suite/autodl_wrapper.log
```

Machine-readable final status is always written to:

```text
${MEDSTYLE_OUTPUT_ROOT}/protocol/autodl_final_status.json
```

After restarting, inspect that file first. It preserves the original suite exit
code and separately records Hugging Face upload/verification, GitHub export, and
shutdown outcomes. If the machine powered off before the shutdown command returned,
the file still contains `shutdown_attempted: true`; use the other fields to decide
whether artifact recovery is required.

Small GitHub files are staged locally under:

```text
${MEDSTYLE_OUTPUT_ROOT}/github_export/
```

The staging manifest is `github_export/export_manifest.json`. GitHub snapshots are
pushed without modifying `main`, using phase-specific branches named:

```text
results/hospital1-<UTC timestamp>
results/hospital2-<UTC timestamp>
```

Their committed files live under `results/final-runs/<run-id>/`. Checkpoints,
Parquet predictions, triplets, audit records, model weights, and other large
artifacts remain Hugging Face-only.
