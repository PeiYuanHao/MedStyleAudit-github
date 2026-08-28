#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEDSTYLE_DATA_ROOT="${MEDSTYLE_DATA_ROOT:-/root/autodl-tmp/datasets}"
export MEDSTYLE_OUTPUT_ROOT="${MEDSTYLE_OUTPUT_ROOT:-/root/autodl-tmp/medstyleaudit-experiments}"
export MEDSTYLE_HF_REPO="${MEDSTYLE_HF_REPO:-PeiyuanHao/MedStyleAudit-Experiments}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface-cache}"

mkdir -p \
  "${MEDSTYLE_DATA_ROOT}/huggingface/Camelyon17-WILDS" \
  "${MEDSTYLE_DATA_ROOT}/camelyon17/images" \
  "${MEDSTYLE_DATA_ROOT}/camelyon17/annotations" \
  "${MEDSTYLE_OUTPUT_ROOT}" \
  "${HF_HOME}"

cd "${REPOSITORY_ROOT}"
python -m pip install -e ".[hf,dev]"
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to start GPU experiments")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

printf '%s\n' \
  "AutoDL directories and Python dependencies are ready." \
  "MEDSTYLE_DATA_ROOT=${MEDSTYLE_DATA_ROOT}" \
  "MEDSTYLE_OUTPUT_ROOT=${MEDSTYLE_OUTPUT_ROOT}" \
  "MEDSTYLE_HF_REPO=${MEDSTYLE_HF_REPO}" \
  "HF_HOME=${HF_HOME}" \
  "No dataset was downloaded."
