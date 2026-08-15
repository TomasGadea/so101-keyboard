#!/usr/bin/env bash
# Start teleoperation: leader (/dev/ttyACM1) drives follower (/dev/ttyACM0).
# Useful for setting the arm up before recording, or sanity-checking hardware.
#
# Usage: bash scripts/teleop.sh
set -euo pipefail

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM1}"

lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id=keyboard_follower \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id=keyboard_leader
