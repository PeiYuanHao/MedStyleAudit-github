#!/usr/bin/env bash
set -euo pipefail

: "${MEDSTYLE_DATA_ROOT:=/workspace/datasets}"
: "${MEDSTYLE_OUTPUT_ROOT:=/workspace/experiments/medstyleaudit}"

mkdir -p \
  "${MEDSTYLE_DATA_ROOT}/huggingface/Camelyon17-WILDS" \
  "${MEDSTYLE_DATA_ROOT}/camelyon17/images" \
  "${MEDSTYLE_DATA_ROOT}/camelyon17/annotations" \
  "${MEDSTYLE_OUTPUT_ROOT}"

printf '%s\n' \
  "Persistent server directories are ready." \
  "Export these variables before running experiments:" \
  "export MEDSTYLE_DATA_ROOT=${MEDSTYLE_DATA_ROOT}" \
  "export MEDSTYLE_OUTPUT_ROOT=${MEDSTYLE_OUTPUT_ROOT}" \
  "" \
  "Dataset download was NOT started. To download WILDS manually:" \
  "hf download wltjr1007/Camelyon17-WILDS --repo-type dataset --local-dir ${MEDSTYLE_DATA_ROOT}/huggingface/Camelyon17-WILDS"
