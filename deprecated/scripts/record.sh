#!/usr/bin/env bash
# Record N teleoperated demonstrations into a LeRobotDataset v3.
# Manual advancement via stdin (n/r/q). See scripts/viz.sh for post-hoc view.
set -euo pipefail

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM1}"
CAMERA="${CAMERA:-/dev/video0}"

NUM_EPISODES="${NUM_EPISODES:-5}"
REPO_ID="${REPO_ID:-local/keyboard-pilot}"
TASK_DESC="${TASK_DESC:-Press the space bar from home pose with pen in gripper}"
FPS="${FPS:-15}"
WARMUP_S="${WARMUP_S:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-3600}"
RESET_TIME_S="${RESET_TIME_S:-3600}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"

DATASET_ROOT="$HOME/.cache/huggingface/lerobot/${REPO_ID}"
if [ -d "${DATASET_ROOT}" ] && [ ! -d "${DATASET_ROOT}/data" ]; then
  rm -rf "${DATASET_ROOT}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/record_with_keys.py" \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id=keyboard_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: ${CAMERA}, width: 640, height: 480, fps: ${FPS}, fourcc: MJPG, warmup_s: ${WARMUP_S}}}" \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id=keyboard_leader \
  --display_data="${DISPLAY_DATA}" \
  --play_sounds=false \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.push_to_hub=false \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.single_task="${TASK_DESC}" \
  --dataset.fps="${FPS}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2
