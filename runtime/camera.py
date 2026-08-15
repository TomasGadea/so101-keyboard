#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.common import load_env_config


def _normalize_camera(camera: str):
    if camera in (None, "", "auto"):
        return "auto"
    if isinstance(camera, str) and camera.startswith("opencv:"):
        camera = camera.split(":", 1)[1]
    if isinstance(camera, str) and camera.isdigit():
        return int(camera)
    return camera


def _opencv_backend():
    if platform.system() == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_V4L2


def _lerobot_backend():
    from lerobot.cameras.configs import Cv2Backends

    if platform.system() == "Darwin":
        return Cv2Backends.AVFOUNDATION
    if platform.system() == "Windows":
        return Cv2Backends.DSHOW
    return Cv2Backends.V4L2


def _candidate_cameras(config: dict) -> list:
    camera = _normalize_camera(config.get("CAMERA", "auto"))
    if camera != "auto":
        return [camera]
    return list(range(4))


def capture_rgb_frame(config: dict) -> np.ndarray:
    """
    Return RGB uint8 image HxWx3.
    Use LeRobot OpenCVCamera with MJPG if available.
    Fall back to cv2.VideoCapture if needed.
    """
    try:
        from lerobot.cameras.configs import ColorMode, Cv2Rotation
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        last_error = None
        for camera_index in _candidate_cameras(config):
            try:
                cfg = OpenCVCameraConfig(
                    index_or_path=camera_index,
                    fps=config["CAMERA_FPS"],
                    width=config["CAMERA_WIDTH"],
                    height=config["CAMERA_HEIGHT"],
                    color_mode=ColorMode.RGB,
                    rotation=Cv2Rotation.NO_ROTATION,
                    fourcc=config["CAMERA_FOURCC"],
                    warmup_s=2,
                    backend=_lerobot_backend(),
                )
                with OpenCVCamera(cfg) as camera:
                    frame = None
                    for _ in range(3):
                        try:
                            frame = camera.async_read(timeout_ms=2000)
                        except TimeoutError:
                            frame = camera.read()
                    if frame is None:
                        frame = camera.read()
                    frame = np.asarray(frame)
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    print(f"[camera] Captured with LeRobot OpenCVCamera({camera_index}).")
                    return frame
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
    except Exception as exc:
        print(f"[camera] LeRobot OpenCVCamera failed: {exc}", file=sys.stderr)
        print("[camera] Falling back to cv2.VideoCapture.", file=sys.stderr)

    last_error = None
    for camera_index in _candidate_cameras(config):
        cap = cv2.VideoCapture(camera_index, _opencv_backend())
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["CAMERA_WIDTH"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["CAMERA_HEIGHT"])
        cap.set(cv2.CAP_PROP_FPS, config["CAMERA_FPS"])
        fourcc = config.get("CAMERA_FOURCC")
        if fourcc and platform.system() != "Darwin":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if not cap.isOpened():
            last_error = RuntimeError(f"Could not open camera {camera_index!r}")
            cap.release()
            continue

        frame_bgr = None
        for _ in range(10):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame_bgr = candidate
                break
            time.sleep(0.05)
        cap.release()
        if frame_bgr is not None:
            print(f"[camera] Captured with cv2.VideoCapture({camera_index}).")
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        last_error = RuntimeError(f"OpenCV camera {camera_index!r} returned no frame.")

    raise RuntimeError(f"Could not capture from camera {config.get('CAMERA')!r}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one RGB camera frame.")
    parser.add_argument("--out", required=True, help="Output image path.")
    parser.add_argument("--camera", default=None, help="Override camera index/path for this run.")
    args = parser.parse_args()

    config = load_env_config()
    if args.camera is not None:
        config["CAMERA"] = args.camera
    print(f"[camera] using CAMERA={config['CAMERA']!r}")
    frame_rgb = capture_rgb_frame(config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    print(f"saved {out_path} ({frame_rgb.shape[1]}x{frame_rgb.shape[0]})")


if __name__ == "__main__":
    main()
