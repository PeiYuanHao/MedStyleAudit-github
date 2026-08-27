#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: MEDSTYLE_RESULT_TAG=my-run bash scripts/autodl_run_export_shutdown.sh <experiment command...>" >&2
  exit 2
fi

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEDSTYLE_DATA_ROOT="${MEDSTYLE_DATA_ROOT:-/root/autodl-tmp/datasets}"
export MEDSTYLE_OUTPUT_ROOT="${MEDSTYLE_OUTPUT_ROOT:-/root/autodl-tmp/medstyleaudit-experiments}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface-cache}"
RESULT_TAG="${MEDSTYLE_RESULT_TAG:-run-$(date -u +%Y%m%dT%H%M%SZ)}"
JOB_LOG="${MEDSTYLE_OUTPUT_ROOT}/autodl-${RESULT_TAG}.log"
JOB_START_EPOCH="$(date +%s)"

mkdir -p "${MEDSTYLE_OUTPUT_ROOT}"
exec > >(tee -a "${JOB_LOG}") 2>&1
cd "${REPOSITORY_ROOT}"

if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "Refusing automatic result push: repository is not on main." >&2
  exit 1
fi
git config user.name >/dev/null 2>&1 || git config user.name "MedStyleAudit AutoDL"
git config user.email >/dev/null 2>&1 || git config user.email "autodl@medstyleaudit.local"

echo "Running experiment command: $*"
"$@"

# Bring main forward before creating the tracked result snapshot. No force push is used.
GIT_TERMINAL_PROMPT=0 git pull --ff-only origin main
python scripts/13_export_results.py \
  --source "${MEDSTYLE_OUTPUT_ROOT}" \
  --repository "${REPOSITORY_ROOT}" \
  --tag "${RESULT_TAG}" \
  --max-file-mb "${MEDSTYLE_GITHUB_MAX_FILE_MB:-20}" \
  --max-total-mb "${MEDSTYLE_GITHUB_MAX_TOTAL_MB:-100}" \
  --since-epoch "${JOB_START_EPOCH}"

git add -- "results/${RESULT_TAG}"
if git diff --cached --quiet; then
  echo "No GitHub-safe result files were produced; refusing to shut down." >&2
  exit 1
fi
git commit -m "results: add AutoDL experiment ${RESULT_TAG}"
GIT_TERMINAL_PROMPT=0 git push origin HEAD:main

echo "Experiment results were pushed to origin/main. AutoDL will now shut down."
if [[ "${MEDSTYLE_SKIP_SHUTDOWN:-0}" == "1" ]]; then
  echo "Shutdown skipped because MEDSTYLE_SKIP_SHUTDOWN=1."
  exit 0
fi
/usr/bin/shutdown
