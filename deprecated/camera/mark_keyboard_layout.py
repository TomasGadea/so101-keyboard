#!/usr/bin/env python3
"""Mark keyboard corners and key centers on a reference image.

Opens a matplotlib window where you:
  1. Click the 4 keyboard corners (top-left, top-right, bottom-right, bottom-left)
  2. Click the center of each key when prompted

Outputs keyboard_layout_qwerty_normalized.json with (u, v) in [0,1] for each key
relative to the keyboard rectangle you defined.

Usage:
    python mark_keyboard_layout.py --image calibration/reference_keyboard.png
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Keys to mark, in order.  Letters row by row, then special keys.
KEYS_TO_MARK: list[str] = [
    # top row
    "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P",
    # home row
    "A", "S", "D", "F", "G", "H", "J", "K", "L",
    # bottom row
    "Z", "X", "C", "V", "B", "N", "M",
    # special
    "SPACE", "ENTER",
]

CORNER_LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_marking_ui(image_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Interactive matplotlib UI. Returns (quad_px [4x2], key_pixels dict)."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    corners: list[list[float]] = []
    key_pixels: dict[str, tuple[float, float]] = {}
    key_queue = list(KEYS_TO_MARK)
    phase = {"current": "corners"}  # "corners" then "keys"
    aborted = {"v": False}

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(image_rgb)
    ax.set_title(f"Click {CORNER_LABELS[0]} corner of the keyboard")
    ax.set_xlabel("Press 'r' to redo last click, 'Esc' to abort")

    corner_line, = ax.plot([], [], "-", color="#ffd400", lw=2)
    corner_pts, = ax.plot([], [], "o", color="red", markersize=10)
    key_pts, = ax.plot([], [], "o", color="#00ff00", markersize=6)
    key_labels: list = []

    def _redraw_corners():
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        corner_pts.set_data(xs, ys)
        if len(corners) == 4:
            corner_line.set_data(xs + [xs[0]], ys + [ys[0]])
        else:
            corner_line.set_data([], [])

    def _redraw_keys():
        xs = [key_pixels[k][0] for k in key_pixels]
        ys = [key_pixels[k][1] for k in key_pixels]
        key_pts.set_data(xs, ys)

    def _update_title():
        if phase["current"] == "corners":
            if len(corners) < 4:
                ax.set_title(f"Click {CORNER_LABELS[len(corners)]} corner of the keyboard")
            else:
                ax.set_title("Corners done — now click key centers")
        elif phase["current"] == "keys":
            if key_queue:
                ax.set_title(f"Click the center of key: {key_queue[0]}  ({len(KEYS_TO_MARK) - len(key_queue)}/{len(KEYS_TO_MARK)})")
            else:
                ax.set_title("All keys marked! Close window or press Enter to finish.")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.xdata is None or event.ydata is None or event.button != 1:
            return
        x, y = float(event.xdata), float(event.ydata)

        if phase["current"] == "corners":
            if len(corners) >= 4:
                return
            corners.append([x, y])
            print(f"  corner {CORNER_LABELS[len(corners) - 1]}: [{x:.1f}, {y:.1f}]")
            _redraw_corners()
            if len(corners) == 4:
                phase["current"] = "keys"
                print("\nCorners set. Now click each key center when prompted.")
            _update_title()

        elif phase["current"] == "keys":
            if not key_queue:
                return
            key = key_queue.pop(0)
            key_pixels[key] = (x, y)
            print(f"  {key}: [{x:.1f}, {y:.1f}]")
            # draw label
            label = ax.annotate(
                key, (x, y), textcoords="offset points", xytext=(5, -5),
                fontsize=7, color="#00ff00", fontweight="bold",
            )
            key_labels.append(label)
            _redraw_keys()
            _update_title()

    def on_key(event):
        if event.key == "escape":
            aborted["v"] = True
            plt.close(fig)
        elif event.key == "r":
            # undo last click
            if phase["current"] == "keys" and key_pixels:
                # undo last key
                last_marked = KEYS_TO_MARK[len(KEYS_TO_MARK) - len(key_queue) - 1]
                del key_pixels[last_marked]
                key_queue.insert(0, last_marked)
                if key_labels:
                    key_labels.pop().remove()
                _redraw_keys()
                print(f"  undid {last_marked}")
            elif phase["current"] == "keys" and not key_pixels:
                # go back to corners
                phase["current"] = "corners"
                corners.pop()
                _redraw_corners()
                print(f"  undid corner {CORNER_LABELS[len(corners)]}")
            elif phase["current"] == "corners" and corners:
                corners.pop()
                _redraw_corners()
                print(f"  undid corner {CORNER_LABELS[len(corners)]}")
            _update_title()
        elif event.key in ("enter", " "):
            if phase["current"] == "keys" and not key_queue:
                plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if aborted["v"]:
        raise KeyboardInterrupt("Aborted by user.")
    if len(corners) != 4:
        raise RuntimeError(f"Need 4 corners, got {len(corners)}")
    if key_queue:
        raise RuntimeError(f"Not all keys marked. Missing: {key_queue}")

    quad = np.array(corners, dtype=np.float32)
    return quad, key_pixels


def compute_normalized_layout(
    quad_px: np.ndarray, key_pixels: dict[str, tuple[float, float]]
) -> dict[str, dict[str, float]]:
    """Convert pixel key positions to normalized (u,v) within the quad."""
    # Homography from quad corners to unit square
    from runtime.geometry import order_quad_points

    ordered = order_quad_points(quad_px)
    unit = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    H_img_to_unit = cv2.getPerspectiveTransform(ordered, unit)

    names = list(key_pixels.keys())
    pts = np.array([key_pixels[n] for n in names], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H_img_to_unit).reshape(-1, 2)

    layout: dict[str, dict[str, float]] = {}
    for name, uv in zip(names, transformed):
        layout[name] = {"u": float(uv[0]), "v": float(uv[1])}
    return layout


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mark keyboard corners and key centers on a reference image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the clean reference image of the keyboard.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: 3d_coordinates/keyboard_layout_qwerty_normalized.json)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.output is None:
        args.output = root / "3d_coordinates" / "keyboard_layout_qwerty_normalized.json"

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"ERROR: Cannot read image: {args.image}", file=sys.stderr)
        return 1
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print(f"Image: {args.image} ({image_rgb.shape[1]}x{image_rgb.shape[0]})")
    print(f"\nStep 1: Click the 4 keyboard corners ({', '.join(CORNER_LABELS)})")
    print(f"Step 2: Click the center of each key when prompted ({len(KEYS_TO_MARK)} keys)")
    print(f"Controls: 'r' = undo last click, 'Esc' = abort, 'Enter' = finish\n")

    quad_px, key_pixels = run_marking_ui(image_rgb)
    layout = compute_normalized_layout(quad_px, key_pixels)

    from runtime.geometry import order_quad_points
    ordered_quad = order_quad_points(quad_px).tolist()

    out = {
        "frame": "keyboard_normalized",
        "description": "QWERTY key centers in normalized keyboard rectangle coordinates.",
        "created_at": _now(),
        "source": "manual_marking",
        "source_image": str(args.image),
        "coordinate_system": {
            "u": "0 = left keyboard edge, 1 = right keyboard edge",
            "v": "0 = top keyboard edge, 1 = bottom keyboard edge",
        },
        "quad_px_used": ordered_quad,
        "keys": layout,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"\nWrote {args.output}")
    print(f"Marked {len(layout)} keys.")
    print(f"\nNext steps:")
    print(f"  python runtime/interactive_calibration.py \\")
    print(f"      --reference-image {args.image} --skip-layout")
    print(f"  python runtime/build_key_targets.py --capture --manual-quad")

    return 0


if __name__ == "__main__":
    sys.exit(main())
