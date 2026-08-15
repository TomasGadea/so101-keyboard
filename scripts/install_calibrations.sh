#!/usr/bin/env bash
# Copy versioned calibration JSONs from this repo into the lerobot cache so
# `lerobot-record`, `lerobot-teleoperate`, etc. can find them by --robot.id.
#
# Run this once after cloning the repo, and re-run if you pull a new calibration.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${HF_LEROBOT_CALIBRATION:-$HOME/.cache/huggingface/lerobot/calibration}"

FOLLOWER_DIR="$CACHE_ROOT/robots/so_follower"
LEADER_DIR="$CACHE_ROOT/teleoperators/so_leader"

mkdir -p "$FOLLOWER_DIR" "$LEADER_DIR"
cp -v "$REPO_DIR/calibration/keyboard_follower.json" "$FOLLOWER_DIR/keyboard_follower.json"
cp -v "$REPO_DIR/calibration/keyboard_leader.json"   "$LEADER_DIR/keyboard_leader.json"

echo "Installed calibrations into $CACHE_ROOT"
