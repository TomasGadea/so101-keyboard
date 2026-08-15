"""Write 3d_coordinates/keyboard_layout_qwerty_normalized.json with a default
QWERTY layout in the keyboard's normalized (u, v) coordinate system, where
(0,0) is the top-left corner of the keyboard body and (1,1) the bottom-right.

The normalized values are templated for a US-layout laptop keyboard. They are
NOT exact for any specific keyboard, so verify the result by inspecting
runtime/debug_overlay.png after running build_key_targets.

Usage:
    python runtime/write_default_layout.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "3d_coordinates" / "keyboard_layout_qwerty_normalized.json"


def default_layout() -> dict[str, dict[str, float]]:
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


def main() -> None:
    layout = default_layout()
    out = {
        "frame": "keyboard_normalized",
        "description": "QWERTY key centers in normalized keyboard rectangle coordinates.",
        "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "source": "hardcoded_default_template",
        "coordinate_system": {
            "u": "0 = left keyboard edge, 1 = right keyboard edge",
            "v": "0 = top keyboard edge, 1 = bottom keyboard edge",
        },
        "keys": layout,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT_PATH} ({len(layout)} keys)")
    print("Verify by running `python runtime/build_key_targets.py --capture` and")
    print("inspecting runtime/debug_overlay.png — keys should land on actual key centers.")


if __name__ == "__main__":
    main()
