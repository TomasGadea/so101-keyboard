#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.visual_type import (
    ARM_JOINTS,
    DEFAULT_HOME_PATH,
    JOINT_LIMITS_DEG,
    TARGET_FRAME,
    WristCamera,
    char_to_key,
    detect_key_pixels,
    draw_debug,
    estimate_image_translation,
    planned_keys,
    resolve_robot_port,
    safe_debug_name,
    validate_calibration_ranges,
)


DEFAULT_URDF = REPO_ROOT / "calibration" / "so101_new_calib.urdf"
DEBUG_DIR = REPO_ROOT / "active_type_debug"


def get_q(robot) -> np.ndarray:
    obs = robot.get_observation()
    return np.array([obs[f"{joint}.pos"] for joint in ARM_JOINTS], dtype=float)


def clamp_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    for i, joint in enumerate(ARM_JOINTS):
        lo, hi = JOINT_LIMITS_DEG[joint]
        q[i] = float(np.clip(q[i], lo, hi))
    return q


def get_xyz(ik: RobotKinematics, robot) -> np.ndarray:
    pose = ik.forward_kinematics(get_q(robot))
    return np.asarray(pose[:3, 3], dtype=float)


def move_joints(robot, target_q: np.ndarray, duration: float) -> None:
    start = get_q(robot)
    target = clamp_q(target_q)
    steps = max(2, int(duration * 50))
    dt = duration / max(steps - 1, 1)
    s = np.linspace(0.0, 1.0, steps)
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    gripper_pos = None
    if "gripper" in robot.bus.motors:
        gripper_pos = float(robot.get_observation()["gripper.pos"])
    for alpha in blend:
        q = start + (target - start) * alpha
        action = {f"{joint}.pos": float(q[i]) for i, joint in enumerate(ARM_JOINTS)}
        if gripper_pos is not None:
            action["gripper.pos"] = gripper_pos
        robot.send_action(action)
        time.sleep(dt)


def move_xyz(robot, ik: RobotKinematics, target_xyz: np.ndarray, duration: float, min_z_m: float) -> None:
    target = np.asarray(target_xyz, dtype=float).reshape(3)
    target[2] = max(target[2], min_z_m)
    current = get_xyz(ik, robot)
    q_seed = get_q(robot)
    steps = max(2, int(duration * 50))
    dt = duration / max(steps - 1, 1)
    s = np.linspace(0.0, 1.0, steps)
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    for waypoint in current + (target - current) * blend[:, None]:
        pose = ik.forward_kinematics(q_seed)
        pose[:3, 3] = waypoint
        for _ in range(12):
            q_seed = ik.inverse_kinematics(q_seed, pose, position_weight=1.0, orientation_weight=0.0)
        action = {f"{joint}.pos": float(q_seed[i]) for i, joint in enumerate(ARM_JOINTS)}
        if "gripper" in robot.bus.motors:
            action["gripper.pos"] = float(robot.get_observation()["gripper.pos"])
        robot.send_action(action)
        time.sleep(dt)


def load_home_q(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    import json

    with open(path) as f:
        data = json.load(f)
    joints = data.get("joints_deg", {})
    if not all(joint in joints for joint in ARM_JOINTS):
        return None
    return np.array([float(joints[joint]) for joint in ARM_JOINTS], dtype=float)


class ActiveTyper:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot = SO101Follower(SOFollowerRobotConfig(port=args.port, id=args.robot_id, use_degrees=True))
        validate_calibration_ranges(self.robot.calibration)
        self.ik = RobotKinematics(args.urdf, TARGET_FRAME, joint_names=ARM_JOINTS)
        self.camera = WristCamera(args.camera, args.camera_width, args.camera_height, args.camera_fps)
        self.press_pixel = np.asarray(args.press_pixel, dtype=float)
        self.debug_dir = Path(args.debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def connect(self) -> None:
        self.robot.connect()

    def disconnect(self) -> None:
        self.robot.disconnect()

    def move_xyz(self, xyz: np.ndarray, duration: float | None = None) -> None:
        move_xyz(self.robot, self.ik, xyz, self.args.move_s if duration is None else duration, self.args.min_z_m)

    def detect_frame(
        self,
        key: str,
        name: str,
        image: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray, dict[str, np.ndarray], np.ndarray]:
        if image is None:
            image = self.camera.capture_rgb()
        key_pixels, _quad, _details = detect_key_pixels(
            image,
            center_dx=self.args.key_center_dx,
            center_dy=self.args.key_center_dy,
            fallback_body_quad=None,
        )
        if key not in key_pixels:
            raise RuntimeError(f"Key {key} not detected; available={sorted(key_pixels)}")
        out = self.debug_dir / f"{safe_debug_name(name)}_overlay.png"
        draw_debug(image, key_pixels, self.press_pixel, out, target_key=key)
        key_px = np.asarray(key_pixels[key], dtype=float)
        error_norm = float(np.linalg.norm(self.press_pixel - key_px))
        return key_px, error_norm, image, key_pixels, np.asarray(_quad, dtype=np.float32)

    def detect(self, key: str, name: str) -> tuple[np.ndarray, float]:
        key_px, error_norm, _image, _key_pixels, _quad = self.detect_frame(key, name)
        return key_px, error_norm

    def evaluate_probe(
        self,
        key: str,
        name: str,
        reference_image: np.ndarray,
        reference_key_pixels: dict[str, np.ndarray],
        reference_quad: np.ndarray,
    ) -> tuple[np.ndarray, float, str]:
        image = self.camera.capture_rgb()
        try:
            key_px, error_norm, _image, _key_pixels, _quad = self.detect_frame(key, name, image=image)
            return key_px, error_norm, "fresh"
        except Exception as fresh_exc:
            try:
                flow = estimate_image_translation(reference_image, image, reference_quad)
                key_pixels = {item_key: np.asarray(px, dtype=float) + flow for item_key, px in reference_key_pixels.items()}
                if key not in key_pixels:
                    raise RuntimeError(f"Key {key} missing from reference detection.")
                out = self.debug_dir / f"{safe_debug_name(name)}_flow_overlay.png"
                draw_debug(image, key_pixels, self.press_pixel, out, target_key=key)
                key_px = np.asarray(key_pixels[key], dtype=float)
                return key_px, float(np.linalg.norm(self.press_pixel - key_px)), f"flow_after_fresh_failed:{fresh_exc}"
            except Exception as flow_exc:
                raw = self.debug_dir / f"{safe_debug_name(name)}_failed_raw.png"
                cv2.imwrite(str(raw), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                raise RuntimeError(f"fresh detection failed ({fresh_exc}); optical flow failed ({flow_exc})") from flow_exc

    def candidate_dirs(self) -> list[np.ndarray]:
        unit = [
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (0.7071, 0.7071),
            (0.7071, -0.7071),
            (-0.7071, 0.7071),
            (-0.7071, -0.7071),
        ]
        return [np.asarray(v, dtype=float) for v in unit]

    def center_key(self, key: str) -> bool:
        step = float(self.args.step_m)
        for iteration in range(self.args.max_iters):
            current_xyz = get_xyz(self.ik, self.robot)
            key_px, current_error, reference_image, reference_key_pixels, reference_quad = self.detect_frame(
                key,
                f"{key}_iter_{iteration:02d}_current",
            )
            print(
                f"[active] {key} iter={iteration} key_px={np.round(key_px, 1).tolist()} "
                f"target={np.round(self.press_pixel, 1).tolist()} |e|={current_error:.1f}px step={step*1000:.1f}mm"
            )
            if current_error <= self.args.tolerance_px:
                print(f"[active] {key} centered within {self.args.tolerance_px:.1f}px")
                return True

            best_dir = None
            best_error = current_error
            for idx, direction in enumerate(self.candidate_dirs()):
                probe_xyz = current_xyz.copy()
                probe_xyz[:2] += direction * step
                self.move_xyz(probe_xyz, duration=self.args.probe_move_s)
                time.sleep(self.args.settle_s)
                try:
                    _probe_px, probe_error, source = self.evaluate_probe(
                        key,
                        f"{key}_iter_{iteration:02d}_probe_{idx:02d}",
                        reference_image,
                        reference_key_pixels,
                        reference_quad,
                    )
                    print(
                        f"[active] probe {idx:02d} dx={direction[0]*step*1000:+.1f}mm "
                        f"dy={direction[1]*step*1000:+.1f}mm -> |e|={probe_error:.1f}px ({source})"
                    )
                    if probe_error < best_error - self.args.min_improve_px:
                        best_error = probe_error
                        best_dir = direction.copy()
                except Exception as exc:
                    print(f"[active] probe {idx:02d} failed: {exc}")
                finally:
                    self.move_xyz(current_xyz, duration=self.args.probe_move_s)
                    time.sleep(self.args.settle_s)

            if best_dir is None:
                step *= 0.5
                print(f"[active] no improving move; shrinking step to {step*1000:.1f}mm")
                if step < self.args.min_step_m:
                    print(f"[active] giving up on {key}: no reliable improving camera move")
                    return False
                continue

            next_xyz = current_xyz.copy()
            next_xyz[:2] += best_dir * step
            print(
                f"[active] keeping dx={best_dir[0]*step*1000:+.1f}mm "
                f"dy={best_dir[1]*step*1000:+.1f}mm, |e| {current_error:.1f}->{best_error:.1f}"
            )
            self.move_xyz(next_xyz, duration=self.args.move_s)
            time.sleep(self.args.settle_s)

        return False

    def press_current_key(self) -> None:
        current = get_xyz(self.ik, self.robot)
        press = current.copy()
        press[2] += self.args.press_delta_z_m
        self.move_xyz(press, duration=self.args.press_s)
        time.sleep(self.args.dwell_s)
        self.move_xyz(current, duration=self.args.retract_s)

    def type_text(self, text: str) -> None:
        if self.args.use_home:
            q = load_home_q(Path(self.args.home_file))
            if q is not None:
                print(f"[home] moving to {self.args.home_file}: {np.round(q, 1).tolist()}")
                move_joints(self.robot, q, duration=self.args.home_move_s)
                time.sleep(self.args.settle_s)
            else:
                print(f"[home] no valid home pose at {self.args.home_file}; using current pose")

        for index, key in enumerate(planned_keys(text), start=1):
            print(f"\n=== {index}/{len(planned_keys(text))} {key} ===")
            ok = self.center_key(key)
            if not ok:
                raise RuntimeError(f"Could not center {key}; see {self.debug_dir}")
            if self.args.press:
                self.press_current_key()
            else:
                print("[active] press disabled; pass --press only after hover alignment is good")


def build_parser() -> argparse.ArgumentParser:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Continuously use the wrist camera to center keys under a fixed press pixel.")
    parser.add_argument("text", nargs="?", default="Q")
    parser.add_argument("--port", default=resolve_robot_port())
    parser.add_argument("--robot-id", default=os.environ.get("ROBOT_ID", "keyboard_follower"))
    parser.add_argument("--urdf", default=os.environ.get("URDF", str(DEFAULT_URDF)))
    parser.add_argument("--camera", default=os.environ.get("CAMERA", "0"))
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--press-pixel", nargs=2, type=float, default=[320.0, 360.0])
    parser.add_argument("--key-center-dx", type=float, default=0.0)
    parser.add_argument("--key-center-dy", type=float, default=0.0)
    parser.add_argument("--step-m", type=float, default=0.008)
    parser.add_argument("--min-step-m", type=float, default=0.00075)
    parser.add_argument("--max-iters", type=int, default=12)
    parser.add_argument("--tolerance-px", type=float, default=18.0)
    parser.add_argument("--min-improve-px", type=float, default=2.0)
    parser.add_argument("--move-s", type=float, default=0.35)
    parser.add_argument("--probe-move-s", type=float, default=0.22)
    parser.add_argument("--settle-s", type=float, default=0.12)
    parser.add_argument("--min-z-m", type=float, default=0.08)
    parser.add_argument("--press", action="store_true", help="Actually press after centering. Default is hover-only.")
    parser.add_argument("--press-delta-z-m", type=float, default=-0.006)
    parser.add_argument("--press-s", type=float, default=0.25)
    parser.add_argument("--retract-s", type=float, default=0.25)
    parser.add_argument("--dwell-s", type=float, default=0.08)
    parser.add_argument("--use-home", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--home-file", default=str(DEFAULT_HOME_PATH))
    parser.add_argument("--home-move-s", type=float, default=1.0)
    parser.add_argument("--debug-dir", default=str(DEBUG_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(f"planned sequence: {' '.join(planned_keys(args.text))}")
    print(f"press pixel: {args.press_pixel}")
    print("mode: continuous camera, fresh detection after every probe, pressing disabled unless --press is set")
    typer = ActiveTyper(args)
    typer.connect()
    try:
        typer.type_text(args.text)
    finally:
        typer.disconnect()


if __name__ == "__main__":
    main()
