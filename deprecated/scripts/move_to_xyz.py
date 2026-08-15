"""Move SO-101 follower to a target (x, y, z) in meters.

Usage:
python scripts/move_to_xyz.py 0.20 0.00 0.10
"""
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

load_dotenv()
PORT = os.environ["PORT"]
ROBOT_ID = "keyboard_follower"
URDF = os.environ["URDF"]
TARGET_FRAME = "gripper"


def main(x: float, y: float, z: float) -> None:
    robot = SO101Follower(SOFollowerRobotConfig(port=PORT, use_degrees=True))
    robot.connect()
    try:
        ik = RobotKinematics(URDF, TARGET_FRAME)
        q_now = np.array([robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors])

        pose = ik.forward_kinematics(q_now)
        pose[:3, 3] = [x, y, z]
        q_target = ik.inverse_kinematics(q_now, pose)

        action = {f"{m}.pos": float(q_target[i]) for i, m in enumerate(robot.bus.motors)}
        action["gripper.pos"] = float(q_now[-1])
        robot.send_action(action)
        time.sleep(2.0)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main(*[float(v) for v in sys.argv[1:4]])
