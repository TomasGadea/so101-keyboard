"""Trace a 1 cm rectangle with the SO-101 end-effector.

Builds on scripts/move_to_xyz.py: each `move_*` call reads the current
joint state, runs forward kinematics, shifts the target by exactly 1 cm in
one axis, and solves inverse kinematics for the new joint angles.

Axis convention (SO-101 base frame):
    move_forward  : +x      move_backward : -x
    move_left     : +y      move_right    : -y
    move_up       : +z      move_down     : -z

Rectangle in the x-y plane: forward, left, backward, right -> back to start.

How to run:
    python scripts/move_in_circle.py
    python scripts/move_in_circle.py --dry-run
"""

import argparse
import os
import time

import numpy as np
from dotenv import load_dotenv

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

load_dotenv()
PORT = os.environ["PORT"]
URDF = os.environ["URDF"]
TARGET_FRAME = "gripper"
STEP = 0.01  # meters per move_* call


class Robot1cm:
    def __init__(self, dry_run: bool = False, settle: float = 0.3):
        self.dry_run = dry_run
        self.settle = settle
        if dry_run:
            self.robot = None
            self.ik = None
            self.target = np.array([0.20, 0.0, 0.10])
            return
        self.robot = SO101Follower(SOFollowerRobotConfig(port=PORT, use_degrees=True))
        self.robot.connect()
        self.ik = RobotKinematics(URDF, TARGET_FRAME)
        q = np.array([self.robot.get_observation()[f"{m}.pos"]
                      for m in self.robot.bus.motors])
        self.target = self.ik.forward_kinematics(q)[:3, 3].copy()

    def _step(self, axis: int, sign: int, name: str) -> None:
        self.target[axis] += sign * STEP
        print(f"{name}()  ->  target=[{self.target[0]:+.3f}, "
              f"{self.target[1]:+.3f}, {self.target[2]:+.3f}]")
        if self.dry_run:
            return
        q_now = np.array([self.robot.get_observation()[f"{m}.pos"]
                          for m in self.robot.bus.motors])
        pose = self.ik.forward_kinematics(q_now)
        pose[:3, 3] = self.target
        q_target = self.ik.inverse_kinematics(q_now, pose)
        action = {f"{m}.pos": float(q_target[i])
                  for i, m in enumerate(self.robot.bus.motors)}
        action["gripper.pos"] = float(q_now[-1])
        self.robot.send_action(action)
        time.sleep(self.settle)

    def move_forward(self):  self._step(0, +1, "move_forward")
    def move_backward(self): self._step(0, -1, "move_backward")
    def move_left(self):     self._step(1, +1, "move_left")
    def move_right(self):    self._step(1, -1, "move_right")
    def move_up(self):       self._step(2, +1, "move_up")
    def move_down(self):     self._step(2, -1, "move_down")

    def disconnect(self):
        if self.robot is not None:
            self.robot.disconnect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--settle", type=float, default=0.3)
    args = p.parse_args()

    bot = Robot1cm(dry_run=args.dry_run, settle=args.settle)
    try:
        print(f"Start position: {bot.target}")
        bot.move_backward()
        bot.move_right()
        bot.move_forward()
        bot.move_left()
        print(f"End position:   {bot.target}")
    finally:
        bot.disconnect()


if __name__ == "__main__":
    main()
