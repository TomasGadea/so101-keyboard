from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def load_env_config() -> dict[str, Any]:
    """
    Load runtime configuration with safe defaults matching calibration.
    """
    load_dotenv()
    root = repo_root()
    urdf = os.environ.get("URDF", "calibration/so101_new_calib.urdf")
    urdf_path = Path(urdf)
    if not urdf_path.is_absolute():
        urdf = str(root / urdf_path)

    return {
        "PORT": os.environ.get("PORT", "/dev/ttyACM0"),
        "URDF": urdf,
        "CAMERA": os.environ.get("CAMERA", "/dev/video0"),
        "CAMERA_WIDTH": _env_int("CAMERA_WIDTH", 640),
        "CAMERA_HEIGHT": _env_int("CAMERA_HEIGHT", 480),
        "CAMERA_FPS": _env_int("CAMERA_FPS", 30),
        "CAMERA_FOURCC": os.environ.get("CAMERA_FOURCC", "MJPG"),
        "TARGET_FRAME": os.environ.get("TARGET_FRAME", "gripper_frame_link"),
    }
