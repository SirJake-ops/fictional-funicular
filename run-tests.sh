#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
TEST_TARGET="LLM_Inference_Engine_Test"
TEST_BINARY="${BUILD_DIR}/tests/${TEST_TARGET}"

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}"
fi

cmake --build "${BUILD_DIR}" --target "${TEST_TARGET}"
"${TEST_BINARY}" "$@"
