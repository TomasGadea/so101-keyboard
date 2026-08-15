#!/usr/bin/env bash
# Eval-2 dry-run wrapper. The real work is in
# new_approach_with_homography/pipeline_eval2.py — single process: one
# camera capture + one Gemini call up front, then N rollouts with reused
# robot/cam/bus/ik connections (mirrors how the actual eval works).
#
#   bash scripts/eval2_random.sh                  # 16 random rollouts
#   bash scripts/eval2_random.sh 8                # 8 random rollouts
#   bash scripts/eval2_random.sh --seed 42        # reproducible random
#   bash scripts/eval2_random.sh a b c d          # specific letters
#   bash scripts/eval2_random.sh --skip-gemini    # reuse last keys.json
#
# Logs go to outputs/eval2/<timestamp>.log alongside the printed summary.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=${LOG_DIR:-outputs/eval2}
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/${TS}.log"

echo "Logging to ${LOG}"
python new_approach_with_homography/pipeline_eval2.py "$@" 2>&1 | tee "${LOG}"
