#!/usr/bin/env bash
set -e

# Run task 1 pipeline from its directory
cd "$(dirname "$0")/new_approach_with_homography"
python pipeline_first_task_single_vlm.py --config pipeline_config_task1.json --ocr-strong --no-hold

