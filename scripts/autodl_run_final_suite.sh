#!/usr/bin/env bash
set -Eeuo pipefail

export MEDSTYLE_DATA_ROOT="${MEDSTYLE_DATA_ROOT:-/root/autodl-tmp/datasets}"
export MEDSTYLE_OUTPUT_ROOT="${MEDSTYLE_OUTPUT_ROOT:-/root/autodl-tmp/medstyleaudit-experiments}"
export MEDSTYLE_HF_REPO="${MEDSTYLE_HF_REPO:-PeiyuanHao/MedStyleAudit-Experiments}"
export MEDSTYLE_SHUTDOWN_ON_FAILURE="${MEDSTYLE_SHUTDOWN_ON_FAILURE:-1}"
: "${HF_TOKEN:?HF_TOKEN must already be present in the environment}"

mkdir -p "${MEDSTYLE_OUTPUT_ROOT}/logs/final_suite"
final_log="${MEDSTYLE_OUTPUT_ROOT}/logs/final_suite/autodl_wrapper.log"
exec > >(tee -a "${final_log}") 2>&1

if [[ "$(git status --porcelain)" != "" ]]; then
  echo "Refusing final execution from a dirty Git checkout."
  exit 2
fi
git pull --ff-only
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${MEDSTYLE_CONDA_ENV:-base}"
fi

set +e
bash scripts/run_final_suite.sh "$@"
suite_rc=$?
set -e

if [[ ${suite_rc} -ne 0 ]]; then
  echo "Final suite failed; preserving local artifacts and attempting a recovery upload."
  if [[ -n "${HF_TOKEN:-}" ]]; then
    python scripts/hf_upload_artifacts.py --root "${MEDSTYLE_OUTPUT_ROOT}" || true
    if ! python scripts/hf_verify_artifacts.py; then
      touch "${MEDSTYLE_OUTPUT_ROOT}/.hf_verification_failed"
    fi
  fi
fi

if [[ -f "${MEDSTYLE_OUTPUT_ROOT}/.hf_verification_failed" ]]; then
  echo "Hugging Face verification failed. Instance will remain running; recover from ${MEDSTYLE_OUTPUT_ROOT}."
  exit ${suite_rc:-1}
fi
if [[ ${suite_rc} -eq 0 || "${MEDSTYLE_SHUTDOWN_ON_FAILURE}" == "1" ]]; then
  echo "Artifacts and state are preserved. Shutting down AutoDL."
  /usr/bin/shutdown
fi
exit ${suite_rc}
