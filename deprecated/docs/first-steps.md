# TA First-Steps Sanity Check

Exact commands to complete the four steps from the TA's onboarding mail
(record → replay → train → deploy). Picks ACT as the BC policy because
it's the simplest image-conditioned option in LeRobot 0.5.1.

Pre-flight (every session, on the laptop):

```bash
# Windows PowerShell (admin) — attach motors + camera into WSL
usbipd attach --wsl --busid 2-1            # follower
usbipd attach --wsl --busid 2-3            # leader
usbipd attach --wsl --busid <camera-busid> # see docs/hardware-wsl.md

# WSL
conda activate lerobot
cd ~/path/to/robot_learning_project
```

If this is a fresh clone, also run the one-time setup in
[`../README.md`](../README.md) (miniforge → env → deps → `install_calibrations.sh`).

---

## Step 1 — Record 20 demonstrations

Same simple motion every time, minimal variation. For us: pen in gripper,
home pose → press space bar → return home.

```bash
# Optional: dry run with a single episode to confirm hardware
NUM_EPISODES=1 REPO_ID=local/keyboard-smoke bash scripts/record.sh

# Full 20-episode run, pushed to a separate repo_id
NUM_EPISODES=20 \
REPO_ID=local/keyboard-20 \
TASK_DESC="Press the space bar from home pose with pen in gripper" \
bash scripts/record.sh
```

During recording (stdin, no X11 needed):

| key | effect |
| --- | --- |
| `n` | end this episode → reset window, then `n` again to start the next |
| `r` | redo current episode |
| `q` | finalize and stop |

Output goes to `~/.cache/huggingface/lerobot/local/keyboard-20/` in
LeRobotDataset v3 format. `record.sh` already sets
`--dataset.streaming_encoding=true` so finalize is automatic when the
last episode ends.

---

## Step 2 — Replay to verify recordings

Keep the keyboard / pen in the exact same position as during recording.
Put the follower in roughly the home pose, then:

```bash
REPO_ID=local/keyboard-20 EPISODE=0 bash scripts/replay.sh
REPO_ID=local/keyboard-20 EPISODE=5 bash scripts/replay.sh
REPO_ID=local/keyboard-20 EPISODE=19 bash scripts/replay.sh
```

(Optional, no hardware) inspect synced camera + joints + actions in rerun:

```bash
REPO_ID=local/keyboard-20 EPISODE=0 bash scripts/viz.sh
```

If a replay misses the key — re-record that episode.

---

## Step 3 — Upload to Brev and train BC (ACT, overfit)

### 3a. Push dataset to the HF Hub (once, from the laptop)

```bash
huggingface-cli login                         # token: write scope
huggingface-cli whoami                        # confirm <HF_USER>

# Re-tag the local dataset under your HF user and push
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/keyboard-20")
ds.push_to_hub("<HF_USER>/keyboard-20", private=True)
PY
```

### 3b. On Brev (GPU instance)

SSH in, then:

```bash
# One-time env setup on the Brev box
git clone <this-repo-url>
cd robot_learning_project
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y -c conda-forge ffmpeg
pip install -r requirements.txt

huggingface-cli login                         # same token
```

Train ACT to overfit on the 20 demos:

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=<HF_USER>/keyboard-20 \
  --output_dir=outputs/train/act_keyboard_20 \
  --job_name=act_keyboard_20 \
  --batch_size=8 \
  --steps=50000 \
  --save_freq=10000 \
  --log_freq=200 \
  --wandb.enable=false
```

Watch the loss go down — for a 20-episode overfit run, expect L1 loss to
drop near zero in 20–50k steps on a single 4090/A100. Last checkpoint
lands in `outputs/train/act_keyboard_20/checkpoints/last/pretrained_model/`.

### 3c. Push checkpoint back to Hub (so the laptop can pull it)

```bash
huggingface-cli upload \
  <HF_USER>/act-keyboard-20 \
  outputs/train/act_keyboard_20/checkpoints/last/pretrained_model
```

---

## Step 4 — Deploy on the robot

Background and object positions must match the recording scene exactly
(same table, same keyboard placement, same camera framing).

On the laptop, with motors + camera attached into WSL:

```bash
huggingface-cli download <HF_USER>/act-keyboard-20 \
  --local-dir checkpoints/act-keyboard-20

# Drive the follower with the trained policy. No leader/teleop — the
# policy is the controller.
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=keyboard_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 15, fourcc: MJPG, warmup_s: 5}}" \
  --policy.path=checkpoints/act-keyboard-20 \
  --policy.device=cuda \
  --display_data=false \
  --play_sounds=false \
  --dataset.repo_id=local/keyboard-eval \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=3 \
  --dataset.single_task="Press the space bar from home pose with pen in gripper" \
  --dataset.fps=15
```

If the laptop has no CUDA GPU, swap `--policy.device=cuda` for `cpu` —
ACT runs on CPU at ~5–10 Hz, which is fine since we're recording at 15 Hz
with a tolerant control loop. Otherwise, deploy on Brev or borrow a
teammate's GPU.

The robot should now reproduce the recorded motion. Inspect the eval
runs:

```bash
REPO_ID=local/keyboard-eval EPISODE=0 bash scripts/viz.sh
```

---

## Done criteria

- [ ] `local/keyboard-20` on disk has 20 episodes, replay matches the keyboard
- [ ] Dataset pushed to `<HF_USER>/keyboard-20`
- [ ] ACT training loss converged on Brev
- [ ] Checkpoint pushed to `<HF_USER>/act-keyboard-20`
- [ ] Robot presses space bar autonomously when policy is loaded
- [ ] Slack update sent before next Thursday session
