#!/usr/bin/env python3
"""Integrated leader-driven calibration for the SO-101 keyboard task.

One process holds:
  - follower port  (default /dev/ttyACM0)
  - leader port    (default /dev/ttyACM1)
  - camera         (default /dev/video0)

A background thread mirrors leader -> follower at ~60 Hz so you can
position the contact point with the leader. When you press ENTER, the
mirror pauses for a single bus cycle, the follower's true joint angles
are read, FK is computed, the camera grabs a frame, and you type the
pixel coordinate from the saved PNG (open it in VS Code and read the
status bar).

After all features are collected, an image -> base XY homography is
fit and written to disk. Optional press-depth calibration follows.

Usage:
  python runtime/teleop_calibration.py
  python runtime/teleop_calibration.py --features KEY_Q,KEY_P,KEY_A,KEY_L,KEY_SPACE
  python runtime/teleop_calibration.py --skip-press

Outputs:
  calibration/keyboard_feature_points.json
  calibration/image_to_base_homography.json
  calibration/press_config.json
  calibration/calibration_session.json
  calibration/calib_images/*.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_FEATURES = [
    "KEY_Q", "KEY_P", "KEY_A", "KEY_L", "KEY_Z", "KEY_M",
    "KEY_SPACE", "KEY_ENTER", "KEY_H",
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

REPO_ROOT = Path(__file__).resolve().parent.parent


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# -----------------------------------------------------------------------------
# Mirror: leader -> follower, with a bus_lock so the main thread can take
# snapshots without colliding with the mirror's own bus traffic.
# -----------------------------------------------------------------------------


class Mirror:
    def __init__(self, follower_port: str, leader_port: str,
                 follower_id: str, leader_id: str, hz: float = 60.0):
        self.hz = hz
        self.follower_port = follower_port
        self.leader_port = leader_port
        self.follower_id = follower_id
        self.leader_id = leader_id

        self.follower = None
        self.leader = None
        self.bus_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.read_errors = 0
        self.write_errors = 0

    def connect(self) -> None:
        from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        self.follower = SO101Follower(SOFollowerRobotConfig(
            port=self.follower_port, id=self.follower_id, use_degrees=True))
        self.leader = SO101Leader(SO101LeaderConfig(
            port=self.leader_port, id=self.leader_id, use_degrees=True))

        print(f"[mirror] connecting follower on {self.follower_port} (id={self.follower_id})")
        self.follower.connect()
        print(f"[mirror] connecting leader on {self.leader_port} (id={self.leader_id})")
        self.leader.connect()

        f_obs = self.follower.get_observation()
        l_act = self.leader.get_action()
        f_q = {k.removesuffix(".pos"): v for k, v in f_obs.items() if k.endswith(".pos")}
        l_q = {k.removesuffix(".pos"): v for k, v in l_act.items()}
        print("[mirror] starting joint check (follower vs leader, deg):")
        for m in f_q:
            if m in l_q:
                print(f"  {m:<14} follower={f_q[m]:+8.2f}  leader={l_q[m]:+8.2f}  Δ={l_q[m]-f_q[m]:+7.2f}")
        max_delta = max(abs(l_q[m] - f_q[m]) for m in f_q if m in l_q)
        if max_delta > 15.0:
            print(f"[mirror] WARNING: leader and follower differ by up to {max_delta:.1f}° on a joint.")
            print("[mirror] The follower will jump to match the leader the moment mirroring starts.")
            print("[mirror] Move the leader to roughly the follower's current pose, then press ENTER.")
            input("[mirror] Press ENTER to start mirroring once you've aligned the leader...")

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="mirror", daemon=True)
        self._thread.start()
        print(f"[mirror] mirror thread started @ {self.hz:.0f} Hz")

    def _loop(self) -> None:
        period = 1.0 / self.hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                with self.bus_lock:
                    action = self.leader.get_action()
                    self.follower.send_action(action)
            except Exception as e:
                self.read_errors += 1
                if self.read_errors % 10 == 1:
                    print(f"[mirror] bus error ({self.read_errors} total): {e}")
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))

    def snapshot_follower_q(self, motor_order: list[str]) -> np.ndarray:
        """Read follower joint angles atomically with the mirror thread."""
        with self.bus_lock:
            obs = self.follower.get_observation()
        return np.array([obs[f"{m}.pos"] for m in motor_order], dtype=float)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[mirror] mirror thread stopped")

    def disconnect(self) -> None:
        self.stop()
        for arm, name in [(self.leader, "leader"), (self.follower, "follower")]:
            if arm is None:
                continue
            try:
                arm.disconnect()
                print(f"[mirror] {name} disconnected")
            except Exception as e:
                print(f"[mirror] {name} disconnect error: {e}")


# -----------------------------------------------------------------------------
# Camera: lerobot OpenCVCamera with the V4L2 backend (default ANY hangs over
# usbipd; see memory/lerobot_camera_v4l2_backend.md).
# -----------------------------------------------------------------------------


class Camera:
    def __init__(self, path: str | int, width: int, height: int, fps: int, fourcc: str):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self._cam = None

    def connect(self) -> None:
        from lerobot.cameras.configs import ColorMode, Cv2Rotation, Cv2Backends
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        cfg = OpenCVCameraConfig(
            index_or_path=self.path, fps=self.fps, width=self.width, height=self.height,
            color_mode=ColorMode.RGB, rotation=Cv2Rotation.NO_ROTATION,
            fourcc=self.fourcc, warmup_s=2,
            backend=Cv2Backends.V4L2,
        )
        self._cam = OpenCVCamera(cfg)
        print(f"[camera] connecting {self.path} ({self.width}x{self.height} {self.fourcc})")
        self._cam.connect()
        print("[camera] connected")

    def grab(self) -> np.ndarray:
        # Discard a stale buffered frame, take the next live one.
        for _ in range(2):
            self._cam.async_read(timeout_ms=2000)
        return np.asarray(self._cam.async_read(timeout_ms=2000))

    def disconnect(self) -> None:
        if self._cam is not None:
            try:
                self._cam.disconnect()
            except Exception:
                pass
            self._cam = None


# -----------------------------------------------------------------------------
# Pixel input: matplotlib in-process click window (mirror thread keeps running
# in background, so the arm stays under leader control while you click).
# Falls back to typed 'u v' if matplotlib has no display backend.
# -----------------------------------------------------------------------------

_MPL_OK: bool | None = None


def _ensure_matplotlib() -> bool:
    """Try to set up an interactive matplotlib backend. Cache the result."""
    global _MPL_OK
    if _MPL_OK is not None:
        return _MPL_OK
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot  # noqa: F401
        import tkinter  # noqa: F401
        _MPL_OK = True
    except Exception as e:
        print(f"[click] matplotlib/Tk unavailable ({e}); falling back to typed entry.")
        _MPL_OK = False
    return _MPL_OK


def parse_uv(text: str) -> tuple[int, int]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("expected 'u v'")
    return int(round(float(parts[0]))), int(round(float(parts[1])))


def _typed_pixel(feature: str, description: str, image_path: Path) -> tuple[int, int]:
    print(f"Open {image_path} in any viewer that shows pixel coords (Paint works).")
    while True:
        ans = input(f"Pixel for {feature} as 'u v': ").strip()
        if not ans:
            continue
        try:
            return parse_uv(ans)
        except ValueError as e:
            print(f"  {e}, retry")


def pick_pixel(feature: str, description: str, image_rgb,
               image_path: Path) -> tuple[int, int]:
    """Click-to-pick. Returns (u, v) once user clicks and closes the window."""
    if not _ensure_matplotlib():
        return _typed_pixel(feature, description, image_path)

    import matplotlib.pyplot as plt

    clicked: dict[str, int | None] = {"u": None, "v": None}

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image_rgb)
    ax.set_title(f"{feature} — click {description}\n(close window when satisfied; press 'r' to reset)")
    ax.set_xlabel(f"saved at {image_path.name}")
    marker = ax.plot([], [], "rx", markersize=18, markeredgewidth=2)[0]

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return
        u, v = int(round(event.xdata)), int(round(event.ydata))
        clicked["u"], clicked["v"] = u, v
        marker.set_data([u], [v])
        ax.set_title(f"{feature} — clicked ({u}, {v}) — close to confirm, click again to redo, 'r' to reset")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "r":
            clicked["u"] = clicked["v"] = None
            marker.set_data([], [])
            ax.set_title(f"{feature} — click {description}")
            fig.canvas.draw_idle()
        elif event.key in ("enter", " "):
            if clicked["u"] is not None:
                plt.close(fig)
        elif event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()  # blocks until window closed; mirror thread keeps running

    if clicked["u"] is None:
        print("[click] no click registered — falling back to typed entry")
        return _typed_pixel(feature, description, image_path)
    return int(clicked["u"]), int(clicked["v"])


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


def prompt_float(message: str, default: float | None = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        ans = input(f"{message}{suffix}: ").strip()
        if not ans and default is not None:
            return float(default)
        try:
            return float(ans)
        except ValueError:
            print("  enter a number, retry")


# -----------------------------------------------------------------------------
# Main calibration flow
# -----------------------------------------------------------------------------


def collect_features(mirror: Mirror, camera: Camera, ik, target_frame: str,
                     features: list[str], images_dir: Path,
                     points_path: Path) -> list[dict[str, Any]]:
    motor_order = list(mirror.follower.bus.motors)
    print(f"[calibration] follower motor order: {motor_order}")

    points: list[dict[str, Any]] = []

    print("\n" + "=" * 78)
    print("KEYBOARD-FEATURE HOMOGRAPHY CALIBRATION")
    print("=" * 78)
    print("For each feature: drive the leader to put the gripper contact point on")
    print("the requested key center, hold the leader still, press ENTER. The script")
    print("snapshots the follower joint angles + a camera frame, then asks for the")
    print("pixel coordinate from the saved PNG.")
    print()

    for i, feature in enumerate(features):
        description = FEATURE_DESCRIPTIONS.get(feature, feature)
        print("-" * 78)
        print(f"Feature {i + 1}/{len(features)}: {feature} — {description}")
        print("-" * 78)

        while True:
            input(f"\nMove follower onto {description}, hold, then press ENTER...")

            try:
                q = mirror.snapshot_follower_q(motor_order)
            except Exception as e:
                print(f"[calibration] follower snapshot failed: {e}")
                if prompt_yes_no("Retry this feature?", default=True):
                    continue
                else:
                    break

            try:
                pose = ik.forward_kinematics(q)
            except Exception as e:
                print(f"[calibration] FK failed: {e}")
                if prompt_yes_no("Retry this feature?", default=True):
                    continue
                else:
                    break
            x, y, z = float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])

            print(f"  joint angles (deg): {dict(zip(motor_order, q.round(2).tolist()))}")
            print(f"  {target_frame} xyz (m): x={x:+.4f}  y={y:+.4f}  z={z:+.4f}")

            print("  capturing camera frame...")
            try:
                frame_rgb = camera.grab()
            except Exception as e:
                print(f"[calibration] camera grab failed: {e}")
                if prompt_yes_no("Retry this feature?", default=True):
                    continue
                else:
                    break
            raw_path = images_dir / f"{feature}_raw_{stamp()}.png"
            cv2.imwrite(str(raw_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            print(f"  saved {raw_path}")

            u, v = pick_pixel(feature, description, frame_rgb, raw_path)
            overlay = frame_rgb.copy()
            cv2.circle(overlay, (u, v), 6, (255, 0, 0), -1)
            cv2.putText(overlay, feature, (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            overlay_path = images_dir / f"{feature}_clicked_{stamp()}.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            print("\n  candidate correspondence:")
            print(f"    pixel:    [{u}, {v}]")
            print(f"    base_xyz: [{x:+.4f}, {y:+.4f}, {z:+.4f}]")

            if prompt_yes_no("  Accept?", default=True):
                points.append({
                    "name": feature,
                    "description": description,
                    "pixel": [int(u), int(v)],
                    "base_xy": [x, y],
                    "base_xyz": [x, y, z],
                    "raw_image": str(raw_path.relative_to(REPO_ROOT)),
                    "clicked_image": str(overlay_path.relative_to(REPO_ROOT)),
                })
                write_json(points_path, points)
                print(f"  saved partial points -> {points_path}")
                break
            if not prompt_yes_no("  Retry this feature?", default=True):
                print(f"  skipping {feature}")
                break

    return points


def fit_homography(points: list[dict[str, Any]], target_frame: str,
                   width: int, height: int, camera_path: str,
                   out_path: Path) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print(f"FITTING IMAGE -> BASE XY HOMOGRAPHY ({len(points)} points)")
    print("=" * 78)

    if len(points) < 4:
        raise RuntimeError(f"need at least 4 correspondences, got {len(points)}")

    image_pts = np.array([p["pixel"] for p in points], dtype=np.float32)
    base_pts = np.array([p["base_xy"] for p in points], dtype=np.float32)

    H, inliers = cv2.findHomography(image_pts, base_pts,
                                    method=cv2.RANSAC,
                                    ransacReprojThreshold=0.005)
    if H is None:
        raise RuntimeError("cv2.findHomography failed")

    pred = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(pred - base_pts, axis=1)
    inlier_flags = (inliers.ravel().astype(bool)
                    if inliers is not None else np.ones(len(points), dtype=bool))

    print("Reprojection errors:")
    for p, e, ok, pp in zip(points, errors, inlier_flags, pred):
        flag = "inlier" if ok else "OUTLIER"
        print(f"  {p['name']:>12s}: {1000 * e:6.2f} mm  {flag}  "
              f"pred=[{pp[0]:+.4f},{pp[1]:+.4f}]  meas=[{p['base_xy'][0]:+.4f},{p['base_xy'][1]:+.4f}]")
    mean_err, max_err = float(errors.mean()), float(errors.max())
    print(f"mean = {1000 * mean_err:.2f} mm   max = {1000 * max_err:.2f} mm")
    if mean_err > 0.004 or max_err > 0.008:
        print("WARNING: error larger than ideal (mean<4mm, max<8mm).")
        if not prompt_yes_no("Save anyway?", default=True):
            raise RuntimeError("homography rejected by user")

    out = {
        "description": "Image pixel -> base XY homography (keyboard features only).",
        "created_at": stamp(),
        "target_frame_used_for_points": target_frame,
        "camera": str(camera_path),
        "image_width": width,
        "image_height": height,
        "H_image_to_base_xy": H.tolist(),
        "mean_error_m": mean_err,
        "max_error_m": max_err,
        "num_points": len(points),
        "points": points,
        "per_point_errors_m": {p["name"]: float(e) for p, e in zip(points, errors)},
        "inliers": {p["name"]: bool(ok) for p, ok in zip(points, inlier_flags)},
    }
    write_json(out_path, out)
    print(f"wrote {out_path}")
    return out


def calibrate_press(mirror: Mirror, ik, target_frame: str,
                    out_path: Path) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print("PRESS-DEPTH CALIBRATION")
    print("=" * 78)
    print("Use the leader to position the gripper contact point.")

    motor_order = list(mirror.follower.bus.motors)

    input("\n[surface] Drive contact point to JUST TOUCH the top of the SPACE key. ENTER when stable...")
    q = mirror.snapshot_follower_q(motor_order)
    pose = ik.forward_kinematics(q)
    surface_z = float(pose[2, 3])
    print(f"  surface z = {surface_z:+.4f} m")

    input("\n[press]   Drive contact point DOWN until SPACE actuates reliably. ENTER when stable...")
    q = mirror.snapshot_follower_q(motor_order)
    pose = ik.forward_kinematics(q)
    press_z = float(pose[2, 3])
    print(f"  press z   = {press_z:+.4f} m")

    measured_offset = press_z - surface_z
    print(f"  measured press offset = {measured_offset:+.4f} m")

    hover = prompt_float("Hover height above surface (m)", default=0.035)
    press_off = prompt_float("Press offset (m, negative = into key)", default=measured_offset)
    dwell = prompt_float("Dwell time while pressing (s)", default=0.12)
    retract = prompt_float("Retract height above surface (m)", default=hover)

    cfg = {
        "description": "Press primitive configuration.",
        "created_at": stamp(),
        "target_frame": target_frame,
        "keyboard_surface_z_m": surface_z,
        "measured_press_z_m": press_z,
        "hover_offset_m": float(hover),
        "press_offset_m": float(press_off),
        "dwell_s": float(dwell),
        "retract_offset_m": float(retract),
        "notes": "press_pose_z = surface_z + press_offset_m; hover_pose_z = surface_z + hover_offset_m",
    }
    write_json(out_path, cfg)
    print(f"wrote {out_path}")
    return cfg


# -----------------------------------------------------------------------------
# CLI / driver
# -----------------------------------------------------------------------------


def parse_features(text: str | None) -> list[str]:
    if not text:
        return list(DEFAULT_FEATURES)
    return [t.strip().upper() for t in text.split(",") if t.strip()]


def parse_camera(text: str) -> str | int:
    if text.startswith("opencv:"):
        text = text.split(":", 1)[1]
    return int(text) if text.isdigit() else text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--follower-port", default=os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0"))
    parser.add_argument("--leader-port", default=os.environ.get("LEADER_PORT", "/dev/ttyACM1"))
    parser.add_argument("--follower-id", default=os.environ.get("FOLLOWER_ID", "keyboard_follower"))
    parser.add_argument("--leader-id", default=os.environ.get("LEADER_ID", "keyboard_leader"))
    parser.add_argument("--urdf", default=os.environ.get("URDF", str(REPO_ROOT / "calibration" / "so101_new_calib.urdf")))
    parser.add_argument("--target-frame", default=os.environ.get("TARGET_FRAME", "gripper_frame_link"))
    parser.add_argument("--camera", default=os.environ.get("CAMERA", "/dev/video0"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--mirror-hz", type=float, default=60.0)
    parser.add_argument("--features", default=None)
    parser.add_argument("--skip-homography", action="store_true")
    parser.add_argument("--skip-press", action="store_true")
    args = parser.parse_args()

    features = parse_features(args.features)
    camera_path = parse_camera(args.camera)

    calib_dir = REPO_ROOT / "calibration"
    images_dir = calib_dir / "calib_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 78)
    print("SO-101 KEYBOARD TASK — INTEGRATED LEADER-DRIVEN CALIBRATION")
    print("#" * 78)
    print(f"follower:    {args.follower_port} (id={args.follower_id})")
    print(f"leader:      {args.leader_port} (id={args.leader_id})")
    print(f"urdf:        {args.urdf}")
    print(f"frame:       {args.target_frame}")
    print(f"camera:      {camera_path} {args.width}x{args.height} @ {args.fps} fps {args.fourcc}")
    print(f"features:    {features}")
    print("#" * 78)

    mirror = Mirror(args.follower_port, args.leader_port,
                    args.follower_id, args.leader_id, hz=args.mirror_hz)
    camera = Camera(camera_path, args.width, args.height, args.fps, args.fourcc)
    ik = None
    points: list[dict[str, Any]] = []
    homography: dict[str, Any] | None = None
    press_cfg: dict[str, Any] | None = None

    try:
        from lerobot.model.kinematics import RobotKinematics

        mirror.connect()
        camera.connect()
        ik = RobotKinematics(args.urdf, args.target_frame)
        mirror.start()

        if not args.skip_homography:
            points = collect_features(mirror, camera, ik, args.target_frame, features,
                                      images_dir, calib_dir / "keyboard_feature_points.json")
            if len(points) >= 4:
                homography = fit_homography(points, args.target_frame,
                                            args.width, args.height, str(camera_path),
                                            calib_dir / "image_to_base_homography.json")
            else:
                print(f"\n[calibration] not enough accepted points ({len(points)} < 4); homography skipped")

        if not args.skip_press:
            if prompt_yes_no("\nRun press-depth calibration now?", default=True):
                press_cfg = calibrate_press(mirror, ik, args.target_frame,
                                            calib_dir / "press_config.json")

        session = {
            "timestamp": stamp(),
            "follower_port": args.follower_port,
            "leader_port": args.leader_port,
            "urdf": args.urdf,
            "target_frame": args.target_frame,
            "camera": str(camera_path),
            "image_size": [args.width, args.height],
            "features": features,
            "num_points": len(points),
            "have_homography": homography is not None,
            "have_press_config": press_cfg is not None,
            "mirror_bus_errors": mirror.read_errors,
        }
        write_json(calib_dir / "calibration_session.json", session)
        print(f"\nSession summary written to {calib_dir / 'calibration_session.json'}")
        print(f"Mirror reported {mirror.read_errors} bus errors during the session.")
        return 0

    except KeyboardInterrupt:
        print("\n[main] interrupted by user")
        return 130
    except Exception as e:
        print(f"\n[main] ERROR: {e}")
        traceback.print_exc()
        return 1
    finally:
        mirror.disconnect()
        camera.disconnect()
        print("[main] done")


if __name__ == "__main__":
    sys.exit(main())
