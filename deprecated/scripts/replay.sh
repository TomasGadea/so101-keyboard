#!/usr/bin/env bash
# Replay a recorded episode through the follower arm — drives the follower to
# the action sequence stored in the dataset. Use this to verify a recording
# captured the intended motion (TA first-steps task #2).
#
# Make sure:
# - the keyboard / objects are in the same position as during recording,
# - the follower is in roughly the same starting pose.
#
# Usage:
#   bash scripts/replay.sh                # episode 0 of local/keyboard-pilot
#   EPISODE=3 bash scripts/replay.sh
#   REPO_ID=local/keyboard-20 EPISODE=7 bash scripts/replay.sh
set -euo pipefail

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
REPO_ID="${REPO_ID:-local/keyboard-pilot}"
EPISODE="${EPISODE:-0}"
FPS="${FPS:-15}"
ROOT="${ROOT:-$HOME/.cache/huggingface/lerobot/${REPO_ID}}"

lerobot-replay \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id=keyboard_follower \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${ROOT}" \
  --dataset.episode="${EPISODE}" \
  --dataset.fps="${FPS}" \
  --play_sounds=false
