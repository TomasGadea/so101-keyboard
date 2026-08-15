#!/usr/bin/env bash
set -e

# Usage: ./task3.sh "hello world"
if [ -z "$1" ]; then
  echo "Usage: $0 \"<sentence to type>\""
  exit 1
fi

# Run task 3 pipeline from its directory, passing the sentence through
cd "$(dirname "$0")/new_approach_with_homography"
python pipeline.py --config pipeline_config_task3.json --no-hold "$@"