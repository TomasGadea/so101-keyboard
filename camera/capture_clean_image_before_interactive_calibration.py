#!/usr/bin/env python3
"""Capture a clean reference image of the keyboard before interactive calibration.

Move the robot arm out of the camera's view, then run this script.
The saved image can be passed to interactive_calibration.py via --reference-image
so that pixel clicks are done on an unobstructed view of the keyboard.

Usage:
    python capture_clean_image_before_interactive_calibration.py \
        --save-reference-image calibration/reference_keyboard.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime.camera import capture_rgb_frame
from runtime.common import load_env_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture a clean keyboard reference image for interactive calibration."
    )
    parser.add_argument(
        "--save-reference-image",
        type=Path,
        default=Path("calibration/reference_keyboard.png"),
        help="Output path for the reference image (default: calibration/reference_keyboard.png)",
    )
    args = parser.parse_args()

    out_path: Path = args.save_reference_image
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Capturing frame from camera...")
    config = load_env_config()
    frame_rgb = capture_rgb_frame(config)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(out_path), frame_bgr)
    print(f"Saved reference image: {out_path}")
    print(
        f"\nNext step:\n"
        f"  python runtime/interactive_calibration.py --reference-image {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
