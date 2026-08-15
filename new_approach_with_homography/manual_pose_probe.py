#!/usr/bin/env python3
"""
Hand-guide the SO101 and print the current gripper_frame_link pose on demand.

How we did it:
- tighten screws on the robot
- 

Torque starts DISABLED so you can drag the arm by hand.

Controls (in the terminal where this script runs):
    <Enter>    print current end-effector position
    t<Enter>   toggle torque on/off
    q<Enter>   quit

(`pynput` is avoided so this works over plain SSH / no display.)
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

load_dotenv()

PORT = os.getenv("PORT", "/dev/ttyACM0")
PROJECT_PATH = os.getenv("PROJECT_PATH",
                         str(Path(__file__).resolve().parents[1]))
URDF = f"{PROJECT_PATH}/new_calibration/so101_new_calib.urdf"
CALIBRATION_DIR = Path(f"{PROJECT_PATH}/new_calibration")
CALIBRATION_ID = "calibration_follower"
TARGET_FRAME = "gripper_frame_link"


def current_pose(robot, ik):
    q = np.array([robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors])
    T = ik.forward_kinematics(q)
    return T, q


def main() -> None:
    robot = SO101Follower(SOFollowerRobotConfig(
        port=PORT,
        id=CALIBRATION_ID,
        calibration_dir=CALIBRATION_DIR,
        use_degrees=True,
    ))
    ik = RobotKinematics(URDF, TARGET_FRAME)

    robot.connect()
    robot.bus.disable_torque()
    torque_on = False
    print("Torque OFF — hand-guide the arm.")
    print("Press <Enter> to read the gripper pose, 't' to toggle torque, 'q' to quit.")

    try:
        while True:
            try:
                cmd = input("> ").strip().lower()
            except EOFError:
                break

            if cmd == "q":
                break
            if cmd == "t":
                if torque_on:
                    robot.bus.disable_torque()
                else:
                    robot.bus.enable_torque()
                torque_on = not torque_on
                print(f"  torque {'ON' if torque_on else 'OFF'}")
                continue

            T, q = current_pose(robot, ik)
            x, y, z = T[:3, 3]
            print(f"  {TARGET_FRAME} position (m): "
                  f"x={x:.4f}, y={y:.4f}, z={z:.4f}")
            print(f"  [{x:.4f}, {y:.4f}, {z:.4f}]")
            print(f"  joints (deg): "
                  + ", ".join(f"{m}={float(q[i]):.2f}"
                              for i, m in enumerate(robot.bus.motors)))
    finally:
        robot.disconnect()
        print("disconnected.")


if __name__ == "__main__":
    main()
