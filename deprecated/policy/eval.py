"""
Roll out the trained ACT policy from `tillwenke/robot_learning_act` on the
real SO-arm for a small number of steps.

Pipeline:
    1. Connect robot + front camera (same setup as take_picture.py).
    2. Move to the same start joint positions used during data collection.
    3. Load the ACT policy from the HF Hub.
    4. Loop `STEPS` times at the dataset's FPS:
         observation -> policy.select_action(...) -> robot.send_action(...)
    5. Disconnect.

Run from inside policy/.venv (requires `lerobot` and a robot connection).

    python eval.py                # default 30 steps @ 15 fps = 2 s
    python eval.py --steps 60     # longer rollout
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

# Reuse robot constants + helpers from the existing pipeline code.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "new_approach_with_homography"))

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(PROJECT_ROOT / ".env")  # picks up PORT, CAMERA_INDEX, PROJECT_PATH

import take_picture  # noqa: E402  -- needs PROJECT_PATH from .env first

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera  # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.cameras.configs import ColorMode, Cv2Rotation  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.robots.so_follower.so_follower import SO100Follower  # noqa: E402
from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig  # noqa: E402


POLICY_REPO = "tillwenke/robot_learning_act"
FPS = 15  # matches the training dataset
STEPS_DEFAULT = 30
CAM_KEY = "observation.images.front"


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_observation(robot, cam, device: str) -> dict:
    """Assemble the (state, image) observation in the format ACT expects."""
    obs = robot.get_observation()
    motors = [f"{m}.pos" for m in robot.bus.motors]
    state = np.array([obs[m] for m in motors], dtype=np.float32)

    frame_rgb = cam.async_read(timeout_ms=2000)  # H x W x 3, uint8, RGB
    img = torch.from_numpy(frame_rgb).to(torch.float32) / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)  # 1 x 3 x H x W

    return {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        CAM_KEY: img.to(device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS_DEFAULT,
                        help=f"Number of policy steps to execute (default: {STEPS_DEFAULT}).")
    parser.add_argument("--repo", default=POLICY_REPO,
                        help="HF Hub repo of the trained policy.")
    args = parser.parse_args()

    device = detect_device()
    print(f"[eval] device: {device}")
    print(f"[eval] loading policy from {args.repo}")
    policy = ACTPolicy.from_pretrained(args.repo)
    policy.to(device)
    policy.eval()
    policy.reset()

    # --- Robot ---
    robot_config = SO100FollowerConfig(
        port=take_picture.PORT,
        id=take_picture.CALIBRATION_ID,
        calibration_dir=take_picture.CALIBRATION_DIR,
    )
    robot = SO100Follower(robot_config)
    robot.connect()

    # --- Camera ---
    cam_cfg = OpenCVCameraConfig(
        index_or_path=take_picture.CAMERA_PATH,
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
        fourcc="MJPG",
        warmup_s=2,
    )
    cam = OpenCVCamera(cam_cfg)
    cam.connect()

    try:
        print("[eval] moving to start position...")
        take_picture.move_to_joint_position(
            robot, joint_positions=take_picture.DEFAULT_POSITIONS, duration=2.0, kp=0.5
        )
        time.sleep(0.5)  # let it settle

        motors = [f"{m}.pos" for m in robot.bus.motors]
        dt = 1.0 / FPS
        print(f"[eval] rolling out {args.steps} steps @ {FPS} fps")
        for step in range(args.steps):
            t0 = time.perf_counter()

            with torch.inference_mode():
                obs_batch = build_observation(robot, cam, device)
                action = policy.select_action(obs_batch)
            action_np = action.squeeze(0).cpu().numpy().astype(np.float32)

            if action_np.shape[0] != len(motors):
                raise RuntimeError(
                    f"Policy action dim {action_np.shape[0]} != robot motor count {len(motors)}"
                )
            cmd = {m: float(v) for m, v in zip(motors, action_np)}
            robot.send_action(cmd)
            print(f"  step {step:>3}: action={action_np.round(2).tolist()}")

            # Match the dataset's control rate.
            sleep = dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

        print("[eval] rollout complete. Holding pose; Ctrl+C to release.")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[eval] interrupted, disconnecting.")
    finally:
        cam.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
