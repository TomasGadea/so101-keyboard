#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.common import load_json, repo_root


def char_to_key(ch: str) -> str:
    if ch == " ":
        return "SPACE"
    if ch.isalpha() and len(ch) == 1:
        return ch.upper()
    raise ValueError(f"Unsupported character {ch!r}; supported input is a-z and space.")


def main() -> None:
    root = repo_root()
    default_targets = root / "runtime" / "current_key_targets_3d.json"

    parser = argparse.ArgumentParser(description="Type text on the current keyboard with SO-101.")
    parser.add_argument("text")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hover-only", action="store_true")
    parser.add_argument(
        "--targets",
        type=Path,
        default=default_targets,
        help=f"Path to key targets JSON (default: {default_targets.relative_to(root)})",
    )
    args = parser.parse_args()

    if not args.targets.exists():
        print(
            f"ERROR: Targets file not found: {args.targets}\n"
            "Run 'python runtime/build_key_targets.py' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    targets = load_json(args.targets)
    key_sequence = [char_to_key(ch) for ch in args.text]
    print("planned sequence:", " ".join(key_sequence))

    missing = [key for key in key_sequence if key not in targets["keys"]]
    if missing:
        raise KeyError(f"Missing keys in target layout: {sorted(set(missing))}")

    if args.dry_run:
        for i, key in enumerate(key_sequence):
            target = targets["keys"][key]
            print(f"{i:02d}: {key:>5s} -> x={target['x']:+.4f} y={target['y']:+.4f} z={target['z']:+.4f} pixel={target['pixel']}")
        print(f"overlay: {root / 'runtime' / 'debug_overlay.png'}")
        return

    from runtime.common import load_env_config
    from runtime.robot_motion import SO101Motion

    config = load_env_config()
    press_cfg = load_json(root / "calibration" / "press_config.json")
    motion = SO101Motion(
        port=config["PORT"],
        urdf=config["URDF"],
        target_frame=config["TARGET_FRAME"],
    )
    motion.connect()
    try:
        for i, key in enumerate(key_sequence):
            target = targets["keys"][key]
            xyz = np.array([target["x"], target["y"], target["z"]], dtype=float)
            print(f"{i + 1}/{len(key_sequence)}: {key} -> {xyz}")
            motion.press_key(xyz, press_cfg, hover_only=args.hover_only)
    finally:
        motion.disconnect()


if __name__ == "__main__":
    main()
