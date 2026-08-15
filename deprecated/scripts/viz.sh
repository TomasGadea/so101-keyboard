#!/usr/bin/env bash
# Open a recorded LeRobotDataset episode in the rerun viewer with all channels
# (camera, joint state, action) on a synced timeline. Post-hoc — does not
# touch any hardware, so it's safe to run with the arms unplugged.
#
# Usage:
#   bash scripts/viz.sh                                # episode 0 of local/keyboard-pilot
#   EPISODE=2 bash scripts/viz.sh
#   REPO_ID=local/keyboard-20 EPISODE=7 bash scripts/viz.sh
set -euo pipefail

REPO_ID="${REPO_ID:-local/keyboard-pilot}"
EPISODE="${EPISODE:-0}"
# Default to the local cache path that record.sh writes to. Pass ROOT="" to
# load a public HF dataset instead (lerobot will download to its cache).
ROOT="${ROOT-$HOME/.cache/huggingface/lerobot/${REPO_ID}}"

echo "Visualizing ${REPO_ID} episode ${EPISODE} from ${ROOT:-HF cache}. Ctrl+C to stop."

if [ -n "${ROOT}" ]; then
  lerobot-dataset-viz --repo-id="${REPO_ID}" --root="${ROOT}" --episode-index="${EPISODE}"
else
  lerobot-dataset-viz --repo-id="${REPO_ID}" --episode-index="${EPISODE}"
fi
