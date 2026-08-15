#!/usr/bin/env bash
set -e

# Usage: ./task2.sh <letter>
if [ -z "$1" ]; then
  echo "Usage: $0 <letter>"
  exit 1
fi

# Run task 2 pipeline from its directory, passing the letter through
cd "$(dirname "$0")/new_approach_with_homography"
python pipeline.py --config pipeline_config_task2.json --ocr-strong --ocr-deadline-s 5.0 --no-hold "$1"
