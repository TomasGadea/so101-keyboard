#!/usr/bin/env python3
"""
Interactive calibration for the SO-101 keyboard task using ONLY:
  - the follower arm,
  - the leader arm / teleop for moving it,
  - the camera,
  - the keyboard.

No printed sheet, colored dots, stickers, stylus, or external calibration target
is required.

This script guides you through:
  1. Keyboard-surface/image calibration:
       image pixel (u, v) -> robot-base keyboard/table-plane coordinate (x, y)
     using visible keyboard features such as Q, P, A, L, Z, M, SPACE, ENTER.

  2. Key surface / press-depth calibration.

  3. Optional normalized QWERTY layout generation from existing keys/corners JSON.

Recommended location:
  runtime/interactive_calibration.py

Typical usage from repo root:
  python runtime/interactive_calibration.py

If you use a separate teleop process to move the follower with the leader arm,
run this mode:
  python runtime/interactive_calibration.py --connect-on-read

Manual fallback without robot FK:
  python runtime/interactive_calibration.py --pose-source manual

Outputs:
  calibration/keyboard_feature_points.json
  calibration/image_to_base_homography.json
  calibration/press_config.json
  calibration/calibration_session.json
  3d_coordinates/keyboard_layout_qwerty_normalized.json
  calibration/calib_images/*.png

Important assumptions:
  - The camera remains fixed after calibration.
  - The keyboard features used for calibration lie approximately on the same
    top/key surface plane.
  - The follower contact point is used consistently. If you type directly with
    the gripper/finger, calibrate using that same physical contact point.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv

DEFAULT_KEYBOARD_FEATURES = [
    "KEY_Q",
    "KEY_P",
    "KEY_A",
    "KEY_L",
    "KEY_Z",
    "KEY_M",
    "KEY_SPACE",
    "KEY_ENTER",
    "KEY_H",
]

FEATURE_DESCRIPTIONS = {
    "KEY_Q": "center of the Q key",
    "KEY_P": "center of the P key",
    "KEY_A": "center of the A key",
    "KEY_L": "center of the L key",
    "KEY_Z": "center of the Z key",
    "KEY_M": "center of the M key",
    "KEY_SPACE": "center of the SPACE bar",
    "KEY_ENTER": "center of the ENTER key",
    "KEY_H": "center of the H key",
    "KEY_R": "center of the R key",
    "KEY_U": "center of the U key",
    "KEY_C": "center of the C key",
    "KEY_N": "center of the N key",
    "CORNER_TOP_LEFT": "top-left outer corner of the keyboard body",
    "CORNER_TOP_RIGHT": "top-right outer corner of the keyboard body",
    "CORNER_BOTTOM_RIGHT": "bottom-right outer corner of the keyboard body",
    "CORNER_BOTTOM_LEFT": "bottom-left outer corner of the keyboard body",
}


# ----------------------------- small utilities -----------------------------


def repo_root_from_this_file() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "runtime":
        return here.parent.parent
    return Path.cwd().resolve()


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def prompt_enter(message: str) -> None:
    print("\n" + message)
    input("Press ENTER when ready...")


def prompt_yes_no(message: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        ans = input(f"{message} {suffix} ").strip().lower()
        if not ans:
            return default
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer y or n.")


def prompt_float(message: str, default: float | None = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        ans = input(f"{message}{suffix}: ").strip()
        if not ans and default is not None:
            return float(default)
        try:
            return float(ans)
        except ValueError:
            print("Please enter a number.")


def parse_xy(text: str) -> tuple[float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("Expected two numbers: x y")
    return float(parts[0]), float(parts[1])


def parse_uv(text: str) -> tuple[int, int]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("Expected two numbers: u v")
    return int(round(float(parts[0]))), int(round(float(parts[1])))


def perspective_transform_points(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out


def parse_feature_list(text: str | None) -> list[str]:
    if text is None or not text.strip():
        return list(DEFAULT_KEYBOARD_FEATURES)
    return [x.strip().upper() for x in text.split(",") if x.strip()]


# ----------------------------- data classes -----------------------------


@dataclass
class Args:
    repo_root: Path
    port: str
    urdf: str
    target_frame: str
    camera: str | int
    camera_width: int
    camera_height: int
    camera_fps: int
    camera_fourcc: str
    features: list[str]
    pose_source: str
    manual_pixels: bool
    connect_on_read: bool
    skip_homography: bool
    skip_press: bool
    skip_layout: bool
    reference_image: str | None


# ----------------------------- robot pose reader -----------------------------


class RobotPoseReader:
    def __init__(
        self,
        port: str,
        urdf: str,
        target_frame: str,
        connect_on_read: bool,
        robot_id: str = "keyboard_follower",
    ):
        self.port = port
        self.urdf = urdf
        self.target_frame = target_frame
        self.connect_on_read = connect_on_read
        self.robot_id = robot_id
        self.robot = None
        self.ik = None

    def _import_robot_classes(self):
        try:
            from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

            return SO101Follower, SOFollowerRobotConfig
        except Exception:
            pass

        try:
            from lerobot.robots.so_follower.so_follower import SO101Follower
            from lerobot.robots.so_follower.config_so_follower import (
                SOFollowerRobotConfig,
            )

            return SO101Follower, SOFollowerRobotConfig
        except Exception:
            pass

        from lerobot.robots.so_follower.so_follower import (
            SO100Follower as SO101Follower,
        )
        from lerobot.robots.so_follower.config_so_follower import (
            SO100FollowerConfig as SOFollowerRobotConfig,
        )

        return SO101Follower, SOFollowerRobotConfig

    def connect(self) -> None:
        if self.robot is not None:
            return

        from lerobot.model.kinematics import RobotKinematics

        SO101Follower, SOFollowerRobotConfig = self._import_robot_classes()

        try:
            cfg = SOFollowerRobotConfig(
                port=self.port, id=self.robot_id, use_degrees=True
            )
        except TypeError:
            try:
                cfg = SOFollowerRobotConfig(port=self.port, id=self.robot_id)
            except TypeError:
                cfg = SOFollowerRobotConfig(port=self.port)

        self.robot = SO101Follower(cfg)
        self.robot.connect()
        self.ik = RobotKinematics(self.urdf, self.target_frame)

    def disconnect(self) -> None:
        if self.robot is not None:
            try:
                self.robot.disconnect()
            finally:
                self.robot = None
                self.ik = None

    def read_pose_xyz(self) -> np.ndarray:
        if self.connect_on_read:
            self.connect()

        if self.robot is None or self.ik is None:
            self.connect()

        obs = self.robot.get_observation()
        motors = list(self.robot.bus.motors)
        q_now = np.array([obs[f"{m}.pos"] for m in motors], dtype=float)
        pose = self.ik.forward_kinematics(q_now)
        xyz = np.asarray(pose[:3, 3], dtype=float)

        if self.connect_on_read:
            self.disconnect()

        return xyz


# ----------------------------- camera capture -----------------------------


class FrameCapture:
    def __init__(
        self, camera: str | int, width: int, height: int, fps: int, fourcc: str
    ):
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc

    def capture(self) -> np.ndarray:
        """Return RGB frame as uint8 HxWx3."""
        try:
            return self._capture_lerobot()
        except Exception as e:
            print(f"[camera] LeRobot camera capture failed: {e}")
            print("[camera] Falling back to raw OpenCV VideoCapture.")
            return self._capture_cv2()

    def _capture_lerobot(self) -> np.ndarray:
        from lerobot.cameras.configs import ColorMode, Cv2Rotation, Cv2Backends
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        kwargs = dict(
            index_or_path=self.camera,
            fps=self.fps,
            width=self.width,
            height=self.height,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
            warmup_s=2,
            backend=Cv2Backends.V4L2,
        )
        if self.fourcc:
            kwargs["fourcc"] = self.fourcc

        try:
            cfg = OpenCVCameraConfig(**kwargs)
        except TypeError:
            kwargs.pop("fourcc", None)
            kwargs.pop("warmup_s", None)
            kwargs.pop("backend", None)
            cfg = OpenCVCameraConfig(**kwargs)

        with OpenCVCamera(cfg) as cam:
            frame = None
            for _ in range(3):
                try:
                    frame = cam.async_read(timeout_ms=2000)
                except TimeoutError:
                    frame = cam.read()
            if frame is None:
                frame = cam.read()
            return np.asarray(frame)

    def _capture_cv2(self) -> np.ndarray:
        cap = cv2.VideoCapture(self.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera!r}")

        frame_bgr = None
        for _ in range(10):
            ok, frame_bgr = cap.read()
            if ok and frame_bgr is not None:
                break
            time.sleep(0.05)
        cap.release()

        if frame_bgr is None:
            raise RuntimeError("OpenCV camera returned no frame.")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# ----------------------------- click UI -----------------------------


class PixelClicker:
    def __init__(self, manual: bool = False):
        self.manual = manual
        self.clicked: list[tuple[int, int]] = []

    def get_one_pixel(
        self, image_rgb: np.ndarray, title: str, feature_name: str
    ) -> tuple[int, int]:
        description = FEATURE_DESCRIPTIONS.get(feature_name, feature_name)

        if self.manual:
            print(f"Click/identify the image pixel for {feature_name}: {description}")
            while True:
                ans = input("Type pixel as 'u v': ").strip()
                try:
                    return parse_uv(ans)
                except ValueError as e:
                    print(e)

        self.clicked = []
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.clicked = [(int(x), int(y))]
                print(f"clicked pixel for {feature_name}: [{x}, {y}]")

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(title, on_mouse)
        print(f"Click {feature_name}: {description}.")
        print("Press ENTER/SPACE to accept, r to reset, q to abort.")

        while True:
            view = image_bgr.copy()
            help_text = f"Click {feature_name}: {description}"
            cv2.putText(
                view,
                help_text[:80],
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            if self.clicked:
                x, y = self.clicked[0]
                cv2.circle(view, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(
                    view,
                    f"({x},{y})",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow(title, view)
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 10, 32):
                if self.clicked:
                    cv2.destroyWindow(title)
                    return self.clicked[0]
                print("No point clicked yet.")
            elif k == ord("r"):
                self.clicked = []
            elif k == ord("q") or k == 27:
                cv2.destroyWindow(title)
                raise KeyboardInterrupt("Pixel click aborted by user.")


# ----------------------------- calibration session -----------------------------


class InteractiveCalibration:
    def __init__(self, args: Args):
        self.args = args
        self.calib_dir = args.repo_root / "calibration"
        self.images_dir = self.calib_dir / "calib_images"
        self.runtime_dir = args.repo_root / "runtime"
        self.coords_dir = args.repo_root / "3d_coordinates"

        self.calib_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.coords_dir.mkdir(parents=True, exist_ok=True)

        self.pose_reader = RobotPoseReader(
            port=args.port,
            urdf=args.urdf,
            target_frame=args.target_frame,
            connect_on_read=args.connect_on_read,
        )
        self.camera = FrameCapture(
            camera=args.camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            fourcc=args.camera_fourcc,
        )
        self.clicker = PixelClicker(manual=args.manual_pixels)

        self.reference_image: np.ndarray | None = None
        if args.reference_image:
            ref_bgr = cv2.imread(args.reference_image)
            if ref_bgr is None:
                raise FileNotFoundError(
                    f"Cannot read reference image: {args.reference_image}"
                )
            self.reference_image = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
            print(f"Loaded reference image: {args.reference_image}")
            print(
                "Pixel clicks will use this clean image instead of live captures.\n"
                "IMPORTANT: Do NOT move the camera or keyboard between taking\n"
                "the reference image and running this calibration."
            )

        self.session: dict[str, Any] = {
            "timestamp": now_stamp(),
            "calibration_mode": "keyboard_features_only",
            "target_frame": args.target_frame,
            "camera": str(args.camera),
            "camera_width": args.camera_width,
            "camera_height": args.camera_height,
            "camera_fps": args.camera_fps,
            "camera_fourcc": args.camera_fourcc,
            "reference_image": args.reference_image,
            "features": args.features,
            "outputs": {},
        }

    def close(self) -> None:
        self.pose_reader.disconnect()

    # ---------------- homography calibration ----------------

    def get_current_xyz(self) -> tuple[float, float, float]:
        if self.args.pose_source == "manual":
            while True:
                ans = input(
                    "Type current robot coordinate as 'x y' in meters: "
                ).strip()
                try:
                    x, y = parse_xy(ans)
                    z = prompt_float("Optional z in meters", default=0.0)
                    return x, y, z
                except ValueError as e:
                    print(e)

        xyz = self.pose_reader.read_pose_xyz()
        print(f"Read {self.args.target_frame} pose:")
        print(f"  x = {xyz[0]:+.6f} m")
        print(f"  y = {xyz[1]:+.6f} m")
        print(f"  z = {xyz[2]:+.6f} m")
        return float(xyz[0]), float(xyz[1]), float(xyz[2])

    def collect_keyboard_feature_points(self) -> list[dict[str, Any]]:
        print("\n" + "=" * 78)
        print("KEYBOARD-FEATURE HOMOGRAPHY CALIBRATION")
        print("=" * 78)
        print(
            "This calibration uses only the keyboard itself.\n\n"
            "For each requested keyboard feature:\n"
            "  1. Use the leader arm / teleop to move the follower contact point\n"
            "     to the requested key center or keyboard corner.\n"
            "  2. Touch lightly; do not push the keyboard around.\n"
            "  3. Press ENTER here.\n"
            "  4. The script reads the robot FK position.\n"
            "  5. The script shows an image and asks you to click that same feature.\n"
            + (
                "     (Using the clean REFERENCE IMAGE — the robot arm will not obstruct the view.)\n\n"
                if self.reference_image is not None
                else "     (Using a LIVE camera frame — the robot arm may obstruct some keys.)\n\n"
            )
            +
            "The default feature set is spread over the keyboard: Q, P, A, L, Z, M,\n"
            "SPACE, ENTER, H. This is usually enough to estimate image -> robot XY\n"
            "over the key surface.\n"
        )

        if self.args.pose_source == "robot":
            if self.args.connect_on_read:
                print(
                    "connect-on-read is enabled. If you are using lerobot-teleoperate,\n"
                    "stop teleop before pressing ENTER here so the robot port is free.\n"
                )
            else:
                print(
                    "The script will keep the robot connection open. Do not run a separate\n"
                    "teleop process at the same time unless your setup supports it.\n"
                )

        points: list[dict[str, Any]] = []

        for i, feature in enumerate(self.args.features):
            description = FEATURE_DESCRIPTIONS.get(feature, feature)
            print("\n" + "-" * 78)
            print(f"Feature {i + 1}/{len(self.args.features)}: {feature}")
            print(f"Target: {description}")
            print("-" * 78)

            while True:
                prompt_enter(
                    f"Move the follower contact point to the {description}.\n"
                    "Use the same physical part of the gripper/finger that will press keys.\n"
                    "Touch lightly and keep the keyboard fixed."
                )

                x, y, z = self.get_current_xyz()

                if self.reference_image is not None:
                    frame_rgb = self.reference_image
                    raw_path = self.images_dir / f"{feature}_ref_{now_stamp()}.png"
                    cv2.imwrite(str(raw_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    print(f"Using reference image (saved copy: {raw_path})")
                else:
                    print("\nCapturing camera frame...")
                    frame_rgb = self.camera.capture()
                    raw_path = self.images_dir / f"{feature}_raw_{now_stamp()}.png"
                    cv2.imwrite(str(raw_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    print(f"Saved raw image: {raw_path}")

                u, v = self.clicker.get_one_pixel(
                    frame_rgb, f"click_{feature}", feature
                )

                overlay = frame_rgb.copy()
                cv2.circle(overlay, (u, v), 6, (255, 0, 0), -1)
                cv2.putText(
                    overlay,
                    feature,
                    (u + 8, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
                overlay_path = self.images_dir / f"{feature}_clicked_{now_stamp()}.png"
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                print("\nCandidate correspondence:")
                print(f"  feature:    {feature} ({description})")
                print(f"  pixel:      [{u}, {v}]")
                print(f"  base_xyz:   [{x:+.6f}, {y:+.6f}, {z:+.6f}]")
                print(f"  image:      {raw_path}")
                print(f"  overlay:    {overlay_path}")

                if prompt_yes_no("Accept this correspondence?", default=True):
                    points.append(
                        {
                            "name": feature,
                            "description": description,
                            "pixel": [int(u), int(v)],
                            "base_xy": [float(x), float(y)],
                            "base_xyz": [float(x), float(y), float(z)],
                            "raw_image": str(raw_path.relative_to(self.args.repo_root)),
                            "clicked_image": str(
                                overlay_path.relative_to(self.args.repo_root)
                            ),
                        }
                    )
                    self.write_json(
                        self.calib_dir / "keyboard_feature_points.json", points
                    )
                    print(
                        f"Saved partial points to {self.calib_dir / 'keyboard_feature_points.json'}"
                    )
                    break

                if not prompt_yes_no("Retry this feature?", default=True):
                    print(f"Skipping {feature}.")
                    break

        if len(points) < 4:
            raise RuntimeError(
                "Need at least 4 accepted correspondences to fit a homography."
            )

        self.write_json(self.calib_dir / "keyboard_feature_points.json", points)
        self.session["outputs"][
            "keyboard_feature_points"
        ] = "calibration/keyboard_feature_points.json"
        return points

    def fit_and_save_homography(self, points: list[dict[str, Any]]) -> dict[str, Any]:
        print("\n" + "=" * 78)
        print("FITTING IMAGE -> BASE XY HOMOGRAPHY")
        print("=" * 78)

        image_pts = np.array([p["pixel"] for p in points], dtype=np.float32)
        base_pts = np.array([p["base_xy"] for p in points], dtype=np.float32)

        H, inliers = cv2.findHomography(
            image_pts,
            base_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=0.005,
        )
        if H is None:
            raise RuntimeError("cv2.findHomography failed.")

        pred = perspective_transform_points(image_pts, H)
        errors = np.linalg.norm(pred - base_pts, axis=1)
        inlier_flags = (
            inliers.ravel().astype(bool)
            if inliers is not None
            else np.ones(len(points), dtype=bool)
        )

        print("Reprojection errors:")
        for p, e, ok, pred_xy in zip(points, errors, inlier_flags, pred):
            status = "inlier" if ok else "OUTLIER"
            print(
                f"  {p['name']:>10s}: {1000 * e:7.2f} mm  {status}  "
                f"pred=[{pred_xy[0]:+.4f}, {pred_xy[1]:+.4f}]  "
                f"meas=[{p['base_xy'][0]:+.4f}, {p['base_xy'][1]:+.4f}]"
            )

        mean_err = float(errors.mean())
        max_err = float(errors.max())
        print(f"\nMean error: {1000 * mean_err:.2f} mm")
        print(f"Max error:  {1000 * max_err:.2f} mm")

        if mean_err > 0.004 or max_err > 0.008:
            print(
                "\nWARNING: error is larger than ideal.\n"
                "Recommended when using keyboard features only: mean < 4 mm and max < 8 mm.\n"
                "Common causes: imprecise touching of key centers, imprecise image clicks,\n"
                "keyboard moved during collection, or inconsistent target frame."
            )
            if not prompt_yes_no("Save this homography anyway?", default=False):
                raise RuntimeError("Homography rejected by user.")

        out = {
            "description": (
                "Homography from image pixel coordinates to robot-base XY coordinates, "
                "calibrated using keyboard features only."
            ),
            "created_at": now_stamp(),
            "calibration_mode": "keyboard_features_only",
            "target_frame_used_for_points": self.args.target_frame,
            "camera": str(self.args.camera),
            "image_width": self.args.camera_width,
            "image_height": self.args.camera_height,
            "reference_image": self.args.reference_image,
            "H_image_to_base_xy": H.tolist(),
            "mean_error_m": mean_err,
            "max_error_m": max_err,
            "num_points": len(points),
            "points": points,
            "per_point_errors_m": {p["name"]: float(e) for p, e in zip(points, errors)},
            "inliers": {p["name"]: bool(ok) for p, ok in zip(points, inlier_flags)},
        }

        out_path = self.calib_dir / "image_to_base_homography.json"
        self.write_json(out_path, out)
        print(f"\nWrote {out_path}")
        self.session["outputs"][
            "image_to_base_homography"
        ] = "calibration/image_to_base_homography.json"
        return out

    # ---------------- press calibration ----------------

    def calibrate_press(self) -> dict[str, Any]:
        print("\n" + "=" * 78)
        print("KEY SURFACE Z / PRESS DEPTH CALIBRATION")
        print("=" * 78)
        print(
            "This step estimates z heights for touching and pressing keys using only\n"
            "the follower and keyboard. Use a large easy key, preferably SPACE.\n"
        )

        prompt_enter(
            "Move the same follower contact point so it JUST TOUCHES the top of SPACE\n"
            "or another large key, without pressing it significantly."
        )
        _, _, surface_z = self.get_current_xyz()
        print(f"Keyboard/key surface z: {surface_z:+.6f} m")

        prompt_enter(
            "Now lower the contact point slowly until the key ACTUATES reliably,\n"
            "without pushing dangerously deep."
        )
        _, _, press_z = self.get_current_xyz()
        print(f"Reliable press z: {press_z:+.6f} m")

        default_press_offset = press_z - surface_z
        print(f"Measured press offset: {default_press_offset:+.6f} m")

        hover_offset = prompt_float(
            "Hover height above key surface in meters", default=0.035
        )
        press_offset = prompt_float(
            "Press offset relative to key surface in meters",
            default=default_press_offset,
        )
        dwell_s = prompt_float("Dwell time while pressing in seconds", default=0.12)
        retract_offset = prompt_float(
            "Retract height above key surface in meters", default=hover_offset
        )

        cfg = {
            "description": "Press primitive configuration for keyboard typing without external tool.",
            "created_at": now_stamp(),
            "target_frame": self.args.target_frame,
            "contact_point": "same follower gripper/finger contact point used during calibration",
            "keyboard_surface_z_m": float(surface_z),
            "measured_press_z_m": float(press_z),
            "hover_offset_m": float(hover_offset),
            "press_offset_m": float(press_offset),
            "dwell_s": float(dwell_s),
            "retract_offset_m": float(retract_offset),
            "notes": (
                "press_pose_z = keyboard_surface_z_m + press_offset_m; "
                "hover_pose_z = keyboard_surface_z_m + hover_offset_m."
            ),
        }

        out_path = self.calib_dir / "press_config.json"
        self.write_json(out_path, cfg)
        print(f"\nWrote {out_path}")
        self.session["outputs"]["press_config"] = "calibration/press_config.json"
        return cfg

    # ---------------- normalized layout ----------------

    def write_normalized_layout(self) -> dict[str, Any]:
        print("\n" + "=" * 78)
        print("NORMALIZED QWERTY LAYOUT")
        print("=" * 78)
        print(
            "This writes 3d_coordinates/keyboard_layout_qwerty_normalized.json.\n"
            "If keys_working.json and corners_working.json exist, A-Z are derived from them.\n"
            "SPACE and ENTER are approximate defaults that you must verify with overlays.\n"
        )

        keys_path = self.coords_dir / "keys_working.json"
        corners_path = self.coords_dir / "corners_working.json"

        if keys_path.exists() and corners_path.exists():
            layout = self._layout_from_existing_files(keys_path, corners_path)
            source = "derived_from_keys_working_and_corners_working"
        else:
            layout = self._default_qwerty_layout()
            source = "hardcoded_default_template"

        layout.setdefault("SPACE", {"u": 0.500, "v": 0.820})
        layout.setdefault("ENTER", {"u": 0.855, "v": 0.505})

        out = {
            "frame": "keyboard_normalized",
            "description": "QWERTY key centers in normalized keyboard rectangle coordinates.",
            "created_at": now_stamp(),
            "source": source,
            "coordinate_system": {
                "u": "0 = left keyboard edge, 1 = right keyboard edge",
                "v": "0 = top keyboard edge, 1 = bottom keyboard edge",
            },
            "keys": layout,
        }

        out_path = self.coords_dir / "keyboard_layout_qwerty_normalized.json"
        self.write_json(out_path, out)
        print(f"Wrote {out_path}")
        print("IMPORTANT: verify SPACE and ENTER in a debug overlay before using them.")
        self.session["outputs"][
            "keyboard_layout_qwerty_normalized"
        ] = "3d_coordinates/keyboard_layout_qwerty_normalized.json"
        return out

    def _coord_scale(self, coord_system: str) -> float:
        if coord_system.startswith("normalized_"):
            return float(coord_system.split("_", 1)[1])
        if coord_system == "normalized":
            return 1.0
        raise ValueError(f"Unsupported coordinate_system: {coord_system!r}")

    def _layout_from_existing_files(
        self, keys_path: Path, corners_path: Path
    ) -> dict[str, dict[str, float]]:
        with open(corners_path) as f:
            corners_data = json.load(f)
        with open(keys_path) as f:
            keys_data = json.load(f)

        cscale = self._coord_scale(corners_data["coordinate_system"])
        kscale = self._coord_scale(keys_data["coordinate_system"])

        coords = corners_data["coordinates"]
        quad_img = np.array(
            [
                [coords["top_left"]["x"] / cscale, coords["top_left"]["y"] / cscale],
                [coords["top_right"]["x"] / cscale, coords["top_right"]["y"] / cscale],
                [
                    coords["bottom_right"]["x"] / cscale,
                    coords["bottom_right"]["y"] / cscale,
                ],
                [
                    coords["bottom_left"]["x"] / cscale,
                    coords["bottom_left"]["y"] / cscale,
                ],
            ],
            dtype=np.float32,
        )

        unit = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        H_img_to_unit = cv2.getPerspectiveTransform(quad_img, unit)

        chars: list[str] = []
        pts_img: list[list[float]] = []
        for item in keys_data["letters"]:
            chars.append(str(item["char"]).upper())
            pts_img.append([float(item["x"]) / kscale, float(item["y"]) / kscale])

        pts_unit = perspective_transform_points(
            np.array(pts_img, dtype=np.float32), H_img_to_unit
        )
        layout: dict[str, dict[str, float]] = {}
        for char, uv in zip(chars, pts_unit):
            layout[char] = {"u": float(uv[0]), "v": float(uv[1])}
        return layout

    def _default_qwerty_layout(self) -> dict[str, dict[str, float]]:
        layout: dict[str, dict[str, float]] = {}
        top = "QWERTYUIOP"
        home = "ASDFGHJKL"
        bottom = "ZXCVBNM"

        for i, c in enumerate(top):
            layout[c] = {"u": 0.145 + i * 0.060, "v": 0.375}
        for i, c in enumerate(home):
            layout[c] = {"u": 0.165 + i * 0.060, "v": 0.505}
        for i, c in enumerate(bottom):
            layout[c] = {"u": 0.195 + i * 0.060, "v": 0.635}

        layout["SPACE"] = {"u": 0.500, "v": 0.820}
        layout["ENTER"] = {"u": 0.855, "v": 0.505}
        return layout

    # ---------------- session save ----------------

    def write_session(self) -> None:
        out_path = self.calib_dir / "calibration_session.json"
        self.write_json(out_path, self.session)
        print(f"\nWrote session summary: {out_path}")

    @staticmethod
    def write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ---------------- driver ----------------

    def run(self) -> None:
        print("\n" + "#" * 78)
        print("SO-101 KEYBOARD TASK — KEYBOARD-ONLY INTERACTIVE CALIBRATION")
        print("#" * 78)
        print(f"Repo root:      {self.args.repo_root}")
        print(f"Robot port:     {self.args.port}")
        print(f"URDF:           {self.args.urdf}")
        print(f"Target frame:   {self.args.target_frame}")
        print(f"Camera:         {self.args.camera}")
        print(f"Image size:     {self.args.camera_width}x{self.args.camera_height}")
        print(f"Pose source:    {self.args.pose_source}")
        print("Features:")
        for f in self.args.features:
            print(f"  - {f}: {FEATURE_DESCRIPTIONS.get(f, f)}")
        print("#" * 78)

        try:
            # Always use connect_on_read for interactive calibration to keep robot passive
            if self.args.pose_source == "robot":
                print("\nNOTE: Robot will connect briefly for each pose read.")
                print("This keeps the robot passive so you can move it manually with teleop.")

            if not self.args.skip_homography:
                points = self.collect_keyboard_feature_points()
                self.fit_and_save_homography(points)
            else:
                print("Skipping homography calibration.")

            if not self.args.skip_press:
                self.calibrate_press()
            else:
                print("Skipping press calibration.")

            if not self.args.skip_layout:
                self.write_normalized_layout()
            else:
                print("Skipping normalized layout generation.")

            print("\n" + "=" * 78)
            print("CALIBRATION COMPLETE")
            print("=" * 78)
            print("Generated files:")
            for _, path in self.session["outputs"].items():
                print(f"  - {path}")
            print(
                "\nNext step: run the key-target builder and inspect the debug overlay before moving the robot."
            )
        finally:
            self.close()
            self.write_session()


# ----------------------------- CLI -----------------------------


def normalize_camera_arg(camera: str) -> str | int:
    if camera.startswith("opencv:"):
        camera = camera.split(":", 1)[1]
    if camera.isdigit():
        return int(camera)
    return camera


def build_args() -> Args:
    load_dotenv()
    root = repo_root_from_this_file()

    parser = argparse.ArgumentParser(
        description="Keyboard-only interactive calibration for SO-101 keyboard typing."
    )
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--port", default=os.environ.get("PORT", "/dev/ttyACM0"))
    parser.add_argument(
        "--urdf",
        default=os.environ.get(
            "URDF", str(root / "calibration" / "so101_new_calib.urdf")
        ),
    )
    parser.add_argument(
        "--target-frame", default=os.environ.get("TARGET_FRAME", "gripper_frame_link")
    )
    parser.add_argument("--camera", default=os.environ.get("CAMERA", "/dev/video0"))
    parser.add_argument(
        "--camera-width", type=int, default=int(os.environ.get("CAMERA_WIDTH", "640"))
    )
    parser.add_argument(
        "--camera-height", type=int, default=int(os.environ.get("CAMERA_HEIGHT", "480"))
    )
    parser.add_argument(
        "--camera-fps", type=int, default=int(os.environ.get("CAMERA_FPS", "30"))
    )
    parser.add_argument(
        "--camera-fourcc", default=os.environ.get("CAMERA_FOURCC", "MJPG")
    )
    parser.add_argument(
        "--features",
        default=None,
        help=(
            "Comma-separated keyboard features to collect. Default: "
            + ",".join(DEFAULT_KEYBOARD_FEATURES)
        ),
    )
    parser.add_argument(
        "--pose-source",
        choices=["robot", "manual"],
        default="robot",
        help="Read base XY from robot FK, or type it manually.",
    )
    parser.add_argument(
        "--manual-pixels",
        action="store_true",
        help="Type pixel coordinates manually instead of using an OpenCV click window.",
    )
    parser.add_argument(
        "--connect-on-read",
        action="store_true",
        help=(
            "Connect to the robot only when reading FK, then disconnect. "
            "Useful if you use lerobot-teleoperate between features."
        ),
    )
    parser.add_argument("--skip-homography", action="store_true")
    parser.add_argument("--skip-press", action="store_true")
    parser.add_argument("--skip-layout", action="store_true")
    parser.add_argument(
        "--reference-image",
        default=None,
        help=(
            "Path to a clean photo of the keyboard taken without the robot in view. "
            "When provided, pixel clicks are done on this image instead of live captures, "
            "so the robot arm does not obstruct the view. The camera and keyboard must not "
            "move between taking this photo and running calibration."
        ),
    )

    ns = parser.parse_args()

    # For interactive calibration, always use connect_on_read mode to keep the robot passive
    # and allow manual movement via teleop during calibration
    connect_on_read = ns.connect_on_read or (ns.pose_source == "robot")

    return Args(
        repo_root=ns.repo_root.resolve(),
        port=ns.port,
        urdf=ns.urdf,
        target_frame=ns.target_frame,
        camera=normalize_camera_arg(ns.camera),
        camera_width=ns.camera_width,
        camera_height=ns.camera_height,
        camera_fps=ns.camera_fps,
        camera_fourcc=ns.camera_fourcc,
        features=parse_feature_list(ns.features),
        pose_source=ns.pose_source,
        manual_pixels=ns.manual_pixels,
        connect_on_read=connect_on_read,
        skip_homography=ns.skip_homography,
        skip_press=ns.skip_press,
        skip_layout=ns.skip_layout,
        reference_image=ns.reference_image,
    )


def main() -> int:
    args = build_args()
    session = InteractiveCalibration(args)
    try:
        session.run()
        return 0
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
