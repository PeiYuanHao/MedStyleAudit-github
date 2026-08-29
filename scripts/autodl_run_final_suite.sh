#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export MEDSTYLE_DATA_ROOT="${MEDSTYLE_DATA_ROOT:-/root/autodl-tmp/datasets}"
export MEDSTYLE_OUTPUT_ROOT="${MEDSTYLE_OUTPUT_ROOT:-/root/autodl-tmp/medstyleaudit-experiments}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface-cache}"
export MEDSTYLE_HF_REPO="${MEDSTYLE_HF_REPO:-PeiyuanHao/MedStyleAudit-Experiments}"
export MEDSTYLE_GITHUB_EXPORT="${MEDSTYLE_GITHUB_EXPORT:-1}"
export MEDSTYLE_GITHUB_MAX_FILE_BYTES="${MEDSTYLE_GITHUB_MAX_FILE_BYTES:-10485760}"

phase=""
devices=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      phase="${2:-}"
      shift 2
      ;;
    --devices)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        devices+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
if [[ "${phase}" != "hospital1" && "${phase}" != "hospital2" ]]; then
  echo "--phase must be hospital1 or hospital2" >&2
  exit 2
fi
if [[ ${#devices[@]} -eq 0 ]]; then
  echo "--devices requires at least one device" >&2
  exit 2
fi

mkdir -p "${MEDSTYLE_OUTPUT_ROOT}/logs/final_suite" "${MEDSTYLE_OUTPUT_ROOT}/protocol"
final_log="${MEDSTYLE_OUTPUT_ROOT}/logs/final_suite/autodl_wrapper.log"
exec > >(tee -a "${final_log}") 2>&1
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
finalizing=0
suite_pid=""

finalize() {
  local suite_rc="$1"
  if [[ ${finalizing} -eq 1 ]]; then
    return 0
  fi
  finalizing=1
  trap - EXIT INT TERM
  set +e
  echo "Finalizing unattended ${phase} run (suite exit code ${suite_rc})."
  python scripts/autodl_unattended.py finalize \
    --phase "${phase}" \
    --suite-exit-code "${suite_rc}" \
    --started-at "${started_at}" \
    --output-root "${MEDSTYLE_OUTPUT_ROOT}"
  finalizer_rc=$?
  if [[ ${finalizer_rc} -ne 0 ]]; then
    echo "Finalizer failed with exit code ${finalizer_rc}; attempting emergency sync and shutdown."
    sync || true
    /usr/bin/shutdown || echo "ERROR: emergency shutdown command failed."
  fi
  return 0
}

on_signal() {
  local signal="$1"
  echo "Received ${signal}; terminating suite and entering finalization."
  if [[ -n "${suite_pid}" ]]; then
    kill -TERM "${suite_pid}" 2>/dev/null || true
    wait "${suite_pid}" 2>/dev/null || true
    suite_pid=""
  fi
  if [[ "${signal}" == "SIGINT" ]]; then
    exit 130
  fi
  exit 143
}

trap 'finalize $?' EXIT
trap 'on_signal SIGINT' INT
trap 'on_signal SIGTERM' TERM

python scripts/autodl_unattended.py preflight

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing final execution from a dirty Git checkout."
  exit 2
fi
git pull --ff-only

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${MEDSTYLE_CONDA_ENV:-base}"
fi

suite_command=(python scripts/run_final_suite.py)
if [[ "${phase}" == "hospital2" ]]; then
  # Authorization only: this never creates or modifies the explicit unlock marker.
  python scripts/autodl_unattended.py authorize-hospital2 --output-root "${MEDSTYLE_OUTPUT_ROOT}"
  suite_command+=(--resume --allow-final-test)
fi
suite_command+=(--devices "${devices[@]}")

set +e
"${suite_command[@]}" &
suite_pid=$!
wait "${suite_pid}"
suite_rc=$?
suite_pid=""
set -e
exit "${suite_rc}"
