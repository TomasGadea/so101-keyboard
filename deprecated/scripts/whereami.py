"""Print current joint angles + end-effector (x, y, z) of the SO-101 follower.
Forward kinematics only — does NOT send any action, safe to run anytime.

Usage: python scripts/whereami.py
"""
import os
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
PORT = os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0")
ROBOT_ID = os.environ.get("ROBOT_ID", "keyboard_follower")
URDF = os.environ.get("URDF", str(REPO_ROOT / "calibration" / "so101_new_calib.urdf"))
TARGET_FRAME = "gripper"


def main() -> None:
    robot = SO101Follower(SOFollowerRobotConfig(port=PORT, id=ROBOT_ID, use_degrees=True))
    robot.connect()
    try:
        obs = robot.get_observation()
        q = np.array([obs[f"{m}.pos"] for m in robot.bus.motors])
        print("Joint angles (deg):")
        for m, v in zip(robot.bus.motors, q):
            print(f"  {m}: {v:+.2f}")

        ik = RobotKinematics(URDF, TARGET_FRAME)
        pose = ik.forward_kinematics(q)
        x, y, z = pose[:3, 3]
        print(f"\nEnd-effector (m):  x={x:+.4f}  y={y:+.4f}  z={z:+.4f}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
