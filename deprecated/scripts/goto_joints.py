"""Smoothly ramp the SO-101 follower to a target joint configuration.
Joint-space only — no IK, no kinematic singularities.

Usage:
  python scripts/goto_joints.py ready        # named preset
  python scripts/goto_joints.py 0 -60 30 70 -90 80 --duration 3.0
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

PORT = os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0")
ROBOT_ID = os.environ.get("ROBOT_ID", "keyboard_follower")
CTRL_HZ = 50

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

PRESETS = {
    # Mid-height, arm forward, well clear of singularities. Gripper open at 80.
    "ready":     [   0.0,  -60.0,  30.0,  70.0,  -90.0,  80.0],
    # Folded up, low torque demand. Useful as a parking pose.
    "stow":      [   0.0, -100.0,  20.0, 100.0, -100.0,  75.0],
}


def quintic(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    f = 10 * s**3 - 15 * s**4 + 6 * s**5
    return start + (end - start) * f[:, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="+",
                        help="Either a preset name or 6 joint angles in deg "
                             "(shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper).")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hold", action="store_true",
                        help="After ramp, keep re-sending the hold command at 50 Hz "
                             "until Ctrl-C. Prevents disconnect from torque-disabling motors.")
    args = parser.parse_args()

    if len(args.targets) == 1 and args.targets[0] in PRESETS:
        target = np.array(PRESETS[args.targets[0]], dtype=float)
        print(f"using preset '{args.targets[0]}' = {target.tolist()}")
    elif len(args.targets) == 6:
        target = np.array([float(v) for v in args.targets])
    else:
        raise SystemExit(f"Need either a preset name ({list(PRESETS)}) or 6 joint values; "
                         f"got {args.targets}")

    robot = SO101Follower(SOFollowerRobotConfig(port=PORT, id=ROBOT_ID, use_degrees=True))
    robot.connect()
    try:
        obs = robot.get_observation()
        start = np.array([obs[f"{m}.pos"] for m in JOINT_ORDER])

        n = max(2, int(args.duration * CTRL_HZ))
        traj = quintic(start, target, n)
        dt = 1.0 / CTRL_HZ
        max_delta = np.max(np.abs(target - start))

        print("joint plan (deg):")
        print(f"  {'joint':<14} {'start':>8} {'end':>8}  {'Δ':>8}")
        for k, name in enumerate(JOINT_ORDER):
            print(f"  {name:<14} {start[k]:+8.2f} {target[k]:+8.2f}  {target[k]-start[k]:+8.2f}")
        print(f"max joint Δ = {max_delta:.2f}°  over {args.duration:.1f}s @ {CTRL_HZ} Hz "
              f"({n} steps)" + ("  [DRY RUN]" if args.dry_run else ""))

        if args.dry_run:
            return

        for q in traj:
            action = {f"{m}.pos": float(q[k]) for k, m in enumerate(JOINT_ORDER)}
            robot.send_action(action)
            time.sleep(dt)

        if args.hold:
            hold = {f"{m}.pos": float(target[k]) for k, m in enumerate(JOINT_ORDER)}
            print(f"\nholding @ {target.tolist()}  —  Ctrl-C to release")
            try:
                while True:
                    robot.send_action(hold)
                    time.sleep(dt)
            except KeyboardInterrupt:
                print("\nreleasing — arm will fall to gravity-rest pose")
        else:
            time.sleep(0.3)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
