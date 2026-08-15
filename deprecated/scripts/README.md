# scripts/

Run from the repo root with `bash scripts/<name>.sh`. All scripts default to
the team's hardware mapping (follower `/dev/ttyACM0`, leader `/dev/ttyACM1`,
camera `/dev/video0`); override with env vars when needed.

## One-time

| Script | What it does |
| --- | --- |
| `install_calibrations.sh` | Copy `calibration/*.json` into the lerobot cache so CLI commands find them via `--robot.id` / `--teleop.id`. |

## Demo recording workflow

| Script | What it does |
| --- | --- |
| `teleop.sh` | Start teleoperation only — leader drives follower. Useful before recording to set up the home pose and verify hardware. |
| `preview_camera.sh` | Continuously snap camera frames to `../snap.jpg`. Open in VS Code (auto-refreshes) to check framing while teleoperating in another terminal. |
| `snap_camera.sh` | One-shot frame grab to a file. |
| `record.sh` | Record N teleoperated episodes (default 5) into a LeRobotDataset v3. No live viewer — see `viz.sh` to inspect the result afterward. |
| `viz.sh` | Open a recorded episode in the rerun viewer (camera + joints + actions on a synced timeline). Post-hoc, hardware-free. |
| `replay.sh` | Drive the follower arm through a recorded episode — confirms the dataset captured the intended motion. |

## Typical first run

```bash
bash scripts/install_calibrations.sh                      # once after clone
bash scripts/teleop.sh                                    # in terminal 1
bash scripts/preview_camera.sh                            # in terminal 2 — adjust camera angle
# Ctrl+C both, then:
bash scripts/record.sh                                    # records 5 episodes
bash scripts/viz.sh                                       # inspect episode 0
EPISODE=0 bash scripts/replay.sh                          # robot replays episode 0
EPISODE=0 REPO_ID=local/keyboard-27 bash scripts/replay.sh
```

When you're satisfied with the dry run, do the full pass:

```bash
NUM_EPISODES=20 REPO_ID=local/keyboard-20 bash scripts/record.sh
```

## Common env vars

| Var | Default | Used by |
| --- | --- | --- |
| `FOLLOWER_PORT` | `/dev/ttyACM0` | record, replay, teleop |
| `LEADER_PORT` | `/dev/ttyACM1` | record, teleop |
| `CAMERA` | `/dev/video0` | record, snap_camera, preview_camera |
| `FPS` | `15` | record, replay |
| `NUM_EPISODES` | `5` | record |
| `REPO_ID` | `local/keyboard-pilot` | record, viz, replay |
| `EPISODE` | `0` | viz, replay |
| `TASK_DESC` | (space-bar text) | record |
| `EPISODE_TIME_S` | `3600` | record (per-episode max; manual advance via stdin keys, this is just a safety net) |
| `RESET_TIME_S` | `3600` | record (between-episode reset window; manual advance via stdin keys) |
| `WARMUP_S` | `5` | record (camera warmup) |
| `DISPLAY_DATA` | `false` | record (set to `true` to enable live rerun viewer — fragile under WSLg software rendering) |

## During recording: keyboard shortcuts

| Key | Effect |
| --- | --- |
| `n` | next phase (end episode, or end reset window) |
| `r` | redo current episode |
| `q` | stop session and finalize |

Pynput's X11 listener doesn't fire in WSL terminals, so `record.sh` runs
through `record_with_keys.py` which reads single keypresses from stdin.
