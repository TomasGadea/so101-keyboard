#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.camera import capture_rgb_frame
from runtime.common import load_env_config, load_json, repo_root, save_json
from runtime.debug_overlay import draw_key_overlay
from runtime.detect_keyboard_quad import detect_keyboard_quad_cv, manual_select_quad, save_quad_json
from runtime.detect_keycaps import fit_qwerty_key_pixels
from runtime.geometry import (
    image_keys_to_base_xy,
    image_to_base_homography_from_file,
    normalized_keys_to_image,
    order_quad_points,
)
from runtime.key_locator import KeyLocator

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CTRL_HZ = 50
BODY_TO_LAYOUT_BOTTOM_FRACTION = 0.68

SCAN_VIEW_PRESETS = [
    # Broad, conservative wrist-camera views. These are deliberately joint-space
    # poses so the camera can look around even before the keyboard target is known.
    [0.0, -60.0, 30.0, 70.0, -90.0, 80.0],
    [25.0, -60.0, 42.0, 70.0, -90.0, 80.0],
    [40.0, -60.0, 48.0, 70.0, -90.0, 80.0],
    [55.0, -60.0, 54.0, 70.0, -90.0, 80.0],
    [-18.0, -60.0, 30.0, 70.0, -90.0, 80.0],
    [18.0, -60.0, 30.0, 70.0, -90.0, 80.0],
    [0.0, -48.0, 20.0, 58.0, -90.0, 80.0],
    [25.0, -48.0, 32.0, 58.0, -90.0, 80.0],
    [40.0, -48.0, 38.0, 58.0, -90.0, 80.0],
    [-15.0, -48.0, 20.0, 58.0, -90.0, 80.0],
    [15.0, -48.0, 20.0, 58.0, -90.0, 80.0],
    [0.0, -72.0, 42.0, 82.0, -90.0, 80.0],
    [25.0, -72.0, 50.0, 82.0, -90.0, 80.0],
    [40.0, -72.0, 56.0, 82.0, -90.0, 80.0],
]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def quintic(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    return start + (end - start) * blend[:, None]


def get_q(robot) -> np.ndarray:
    obs = robot.get_observation()
    return np.array([obs[f"{m}.pos"] for m in JOINT_ORDER], dtype=float)


def move_joints(robot, target: np.ndarray, duration: float = 1.2) -> None:
    start = get_q(robot)
    n = max(2, int(duration * CTRL_HZ))
    dt = duration / max(n - 1, 1)
    for q in quintic(start, target, n):
        robot.send_action({f"{m}.pos": float(q[i]) for i, m in enumerate(JOINT_ORDER)})
        time.sleep(dt)


def capture_scan_frame(config: dict, frame_path: Path) -> np.ndarray:
    image_rgb = capture_rgb_frame(config)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return image_rgb


def draw_scan_quad(image_rgb: np.ndarray, quad: np.ndarray, out_path: Path) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    q = order_quad_points(quad).astype(np.int32)
    cv2.polylines(image_bgr, [q], True, (0, 255, 255), 3)
    for idx, (x, y) in enumerate(q):
        cv2.circle(image_bgr, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(image_bgr, str(idx), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(out_path), image_bgr)


def keyboard_body_to_layout_quad(body_quad: np.ndarray) -> np.ndarray:
    """
    Convert a detected physical keyboard body into the layout rectangle.

    The normalized key layout was marked on the active key area, not on the
    entire keyboard shell. Logitech boards in these views have a large palm
    rest, so using the body bottom directly projects lower-row keys too low.
    """
    q = order_quad_points(body_quad).astype(np.float32)
    top_left, top_right, bottom_right, bottom_left = q
    layout_bottom_left = top_left + BODY_TO_LAYOUT_BOTTOM_FRACTION * (bottom_left - top_left)
    layout_bottom_right = top_right + BODY_TO_LAYOUT_BOTTOM_FRACTION * (bottom_right - top_right)
    return np.vstack([top_left, top_right, layout_bottom_right, layout_bottom_left]).astype(np.float32)


def scan_view_selection_score(confidence: float, layout_quad: np.ndarray, image_shape: tuple[int, ...]) -> float:
    h, w = image_shape[:2]
    q = order_quad_points(layout_quad).astype(np.float32)
    offscreen = 0.0
    for x, y in q:
        offscreen += max(0.0, -float(x)) + max(0.0, float(x) - (w - 1))
        offscreen += max(0.0, -float(y)) + max(0.0, float(y) - (h - 1))
    offscreen_norm = offscreen / float(w + h)
    border_vertices = sum(
        x <= 2 or x >= w - 3 or y <= 2 or y >= h - 3
        for x, y in q
    )
    return float(confidence - 0.8 * offscreen_norm - 0.035 * border_vertices)


def keycap_selection_score(key_fit_details: dict) -> float:
    rows = key_fit_details.get("fitted_rows", {})
    if not rows:
        return 0.0
    matched = sum(float(row.get("matched", 0.0)) for row in rows.values())
    residuals = [float(row.get("residual", 20.0)) for row in rows.values()]
    dxs = [float(row.get("dx", 0.0)) for row in rows.values() if row.get("dx", 0.0)]
    residual = float(np.mean(residuals)) if residuals else 20.0
    dx_consistency = 1.0
    if len(dxs) >= 2:
        dx_consistency = max(0.0, 1.0 - float(np.std(dxs)) / max(float(np.mean(dxs)), 1e-6))
    return float((matched / 26.0) + 0.18 * dx_consistency - residual / 35.0)


def capture_best_scan_view(config: dict, duration: float = 1.2) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Move the wrist camera through view poses and return the best keyboard quad.
    """
    from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

    root = repo_root()
    scan_dir = root / "runtime" / "scan_frames"
    scan_dir.mkdir(parents=True, exist_ok=True)

    robot = SO101Follower(SOFollowerRobotConfig(port=config["PORT"], id="keyboard_follower", use_degrees=True))
    robot.connect()
    best: tuple[float, float, np.ndarray, np.ndarray, Path] | None = None

    try:
        start_q = get_q(robot)
        candidates = [start_q] + [np.array(preset, dtype=float) for preset in SCAN_VIEW_PRESETS]

        for idx, target_q in enumerate(candidates):
            print(f"[scan] view {idx + 1}/{len(candidates)} target joints: {np.round(target_q, 1).tolist()}")
            move_joints(robot, target_q, duration=duration)
            time.sleep(0.25)

            frame_path = scan_dir / f"view_{idx:02d}.png"
            image_rgb = capture_scan_frame(config, frame_path)
            try:
                quad, confidence, details = detect_keyboard_quad_cv(image_rgb)
                print(f"[scan] detected keyboard in {frame_path.name}: confidence={confidence:.3f}, {details}")
                if details.get("detector") == "dark_body":
                    quad = keyboard_body_to_layout_quad(quad)
                draw_scan_quad(image_rgb, quad, scan_dir / f"view_{idx:02d}_quad.png")
                selection_score = scan_view_selection_score(confidence, quad, image_rgb.shape)
                try:
                    _key_pixels, key_fit_details = fit_qwerty_key_pixels(image_rgb, quad)
                    fit_score = keycap_selection_score(key_fit_details)
                    selection_score = 0.15 * selection_score + fit_score
                    print(f"[scan] keycap fit score for {frame_path.name}: {fit_score:.3f}")
                except Exception as fit_exc:
                    selection_score -= 0.18
                    print(f"[scan] keycap fit failed in {frame_path.name}: {fit_exc}")
                print(f"[scan] selection score for {frame_path.name}: {selection_score:.3f}")
            except Exception as exc:
                print(f"[scan] no keyboard quad in {frame_path.name}: {exc}")
                continue

            if best is None or selection_score > best[0]:
                best = (selection_score, confidence, image_rgb, quad, frame_path)

        move_joints(robot, start_q, duration=duration)
    finally:
        robot.disconnect()

    if best is None:
        raise RuntimeError(
            "Could not detect keyboard boundaries from any scan view. "
            f"Saved scan images in {scan_dir}; inspect them to see what the wrist camera sees."
        )

    selection_score, confidence, image_rgb, quad, frame_path = best
    print(f"[scan] using {frame_path.name} with confidence={confidence:.3f}, selection_score={selection_score:.3f}")
    return image_rgb, quad, confidence


def build_key_targets(
    image_rgb: np.ndarray,
    layout_path: Path,
    homography_path: Path,
    press_config_path: Path,
    quad_px: np.ndarray | None = None,
    manual_quad: bool = False,
    layout_orientation: str = "flip-uv",
    target_source: str = "vlm",
) -> dict:
    """
    Returns dictionary with keys[key] = {"x": ..., "y": ..., "z": ..., "pixel": [u,v]}.

    target_source:
      "vlm"     — KeyLocator: Gemini flash-lite (first call) → ORB warm cache.
                  ~50 ms warm, ~2-3 s cold. RECOMMENDED.
      "keycaps" — Classical CV: detect_keyboard_quad + connected-component fit.
      "quad"    — Project a normalized layout JSON through the detected quad.
    """
    root = repo_root()
    runtime_dir = root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    frame_path = runtime_dir / "current_frame.png"
    cv2.imwrite(str(frame_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    press_cfg = load_json(press_config_path)
    H_image_to_base = image_to_base_homography_from_file(homography_path)
    key_fit_details = None

    if target_source == "vlm":
        # No quad detection needed — KeyLocator finds the keyboard internally
        # via ORB matching against its own reference cache, falling back to
        # Gemini on a cache miss. The "quad" we save is just the Q/P/Z/M
        # anchor box for compatibility with the existing JSON schema.
        try:
            locator = KeyLocator()
            key_pixels_tuples, src, dt = locator.locate(image_rgb)
            key_pixels = {k: np.array(v, dtype=np.float64)
                          for k, v in key_pixels_tuples.items()}
            key_fit_details = {"method": f"key_locator/{src}",
                               "elapsed_s": round(dt, 3),
                               "n_keys": len(key_pixels)}
            print(f"[vlm] KeyLocator: source={src}  dt={dt*1000:.0f}ms  "
                  f"keys={len(key_pixels)}")
            # Synthesize a quad from the Q/P/Z/M anchors. Saved for the
            # existing JSON schema + debug overlay; not used downstream.
            from runtime.key_locator import ANCHORS as _ANCHORS  # noqa: F401
            ref_json = runtime_dir / "keyboard_reference" / "reference_layout.json"
            if ref_json.exists():
                ref = json.loads(ref_json.read_text())
                if "anchors" in ref:
                    a = ref["anchors"]
                    quad_px = np.array([a["Q"], a["P"], a["M"], a["Z"]],
                                       dtype=np.float32)
            if quad_px is None:
                # First time: derive a synthetic quad from the projected layout.
                quad_px = np.array([key_pixels["Q"], key_pixels["P"],
                                    key_pixels["M"], key_pixels["Z"]],
                                   dtype=np.float32)
            confidence = 1.0
            method = "vlm_anchors"
        except Exception as exc:
            print(f"[vlm] KeyLocator failed ({exc}); falling back to keycaps path")
            target_source = "keycaps"

    if target_source != "vlm":
        # Legacy path: quad detection + connected-component keycap fit.
        if quad_px is None:
            if manual_quad:
                quad_px = manual_select_quad(image_rgb)
                confidence = 1.0
                method = "manual"
            else:
                quad_px, confidence, details = detect_keyboard_quad_cv(image_rgb)
                if details.get("detector") == "dark_body":
                    quad_px = keyboard_body_to_layout_quad(quad_px)
                method = "cv_contour"
        else:
            quad_px = order_quad_points(quad_px)
            confidence = 1.0
            method = "provided"

    quad_path = runtime_dir / "current_keyboard_quad.json"
    quad_data = save_quad_json(quad_path, quad_px, confidence, method, image_rgb.shape)
    ordered_quad = np.asarray(quad_data["quad_px"], dtype=np.float32)

    layout = oriented_layout(load_json(layout_path), layout_orientation)
    if target_source == "keycaps":
        try:
            key_pixels, key_fit_details = fit_qwerty_key_pixels(image_rgb, ordered_quad)
            print(
                "[keycaps] fitted QWERTY rows from legends: "
                f"{key_fit_details['fitted_rows']}"
            )
        except Exception as exc:
            print(f"[keycaps] fit failed, falling back to quad layout projection: {exc}")
            key_pixels = normalized_keys_to_image(layout, ordered_quad)
            target_source = "quad_fallback"
    elif target_source == "quad":
        key_pixels = normalized_keys_to_image(layout, ordered_quad)
    elif target_source != "vlm":
        raise ValueError(f"Unsupported target source: {target_source}")

    key_base_xy = image_keys_to_base_xy(key_pixels, H_image_to_base)
    surface_z = float(press_cfg["keyboard_surface_z_m"])

    keys: dict[str, dict] = {}
    for key in sorted(key_pixels.keys()):
        px = key_pixels[key]
        xy = key_base_xy[key]
        keys[key] = {
            "x": float(xy[0]),
            "y": float(xy[1]),
            "z": surface_z,
            "pixel": [int(round(float(px[0]))), int(round(float(px[1])))],
        }

    targets = {
        "frame": "robot_base",
        "source": {
            "image": _rel(frame_path),
            "layout": _rel(layout_path),
            "homography": _rel(homography_path),
            "press_config": _rel(press_config_path),
            "layout_orientation": layout_orientation,
            "target_source": target_source,
        },
        "keyboard_quad_px": quad_data["quad_px"],
        "keys": keys,
    }
    if key_fit_details is not None:
        targets["keycap_fit"] = key_fit_details
    save_json(runtime_dir / "current_key_targets_3d.json", targets)
    draw_key_overlay(image_rgb, ordered_quad, key_pixels, runtime_dir / "debug_overlay.png", show_all=True)
    return targets


def oriented_layout(layout: dict, orientation: str) -> dict:
    if orientation not in {"identity", "flip-u", "flip-v", "flip-uv"}:
        raise ValueError(f"Unsupported layout orientation: {orientation}")

    out = copy.deepcopy(layout)
    flip_u = orientation in {"flip-u", "flip-uv"}
    flip_v = orientation in {"flip-v", "flip-uv"}
    for key in out.get("keys", {}).values():
        if flip_u:
            key["u"] = 1.0 - float(key["u"])
        if flip_v:
            key["v"] = 1.0 - float(key["v"])
    return out


def main() -> None:
    root = repo_root()
    default_ref_image = root / "calibration" / "reference_keyboard.png"
    parser = argparse.ArgumentParser(description="Build current 3D key targets.")
    parser.add_argument(
        "--image",
        default=None,
        help=(
            "Path to an image of the keyboard. If omitted, uses the reference image "
            f"at {_rel(default_ref_image)} when it exists, otherwise captures live."
        ),
    )
    parser.add_argument("--capture", action="store_true", help="Force a live camera capture.")
    parser.add_argument(
        "--capture-scan",
        action="store_true",
        help="Move the robot through wrist-camera view poses and use the best automatic keyboard detection.",
    )
    parser.add_argument("--manual-quad", action="store_true", help="Manually click 4 keyboard corners.")
    parser.add_argument("--auto-quad", action="store_true", help="Force automatic keyboard boundary detection.")
    parser.add_argument(
        "--target-source",
        choices=("vlm", "keycaps", "quad"),
        default="vlm",
        help=("vlm: KeyLocator (Gemini + ORB cache, recommended). "
              "keycaps: classical CV connected-component fit. "
              "quad: normalized layout warped through detected quad."),
    )
    parser.add_argument("--scan-duration", type=float, default=1.2, help="Seconds per scan pose movement.")
    parser.add_argument("--layout", type=Path, default=root / "3d_coordinates" / "keyboard_layout_qwerty_normalized.json")
    parser.add_argument("--homography", type=Path, default=root / "calibration" / "image_to_base_homography.json")
    parser.add_argument("--press-config", type=Path, default=root / "calibration" / "press_config.json")
    parser.add_argument(
        "--layout-orientation",
        choices=("identity", "flip-u", "flip-v", "flip-uv"),
        default="flip-uv",
        help="Transform normalized layout into the current wrist-camera keyboard orientation.",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Override camera index/path for this run, e.g. 0 for the wrist camera.",
    )
    args = parser.parse_args()

    if args.capture_scan and args.manual_quad:
        raise ValueError("--capture-scan is automatic; do not combine it with --manual-quad.")

    config = load_env_config()
    if args.camera is not None:
        config["CAMERA"] = args.camera
    print(f"[camera] using CAMERA={config['CAMERA']!r}")

    # --- resolve image ---
    scan_quad_px = None
    if args.capture_scan:
        image_rgb, scan_quad_px, scan_confidence = capture_best_scan_view(config, duration=args.scan_duration)
        print(f"[scan] best automatic quad confidence: {scan_confidence:.3f}")
    elif args.capture:
        image_rgb = capture_rgb_frame(config)
    elif args.image:
        image_bgr = cv2.imread(args.image)
        if image_bgr is None:
            raise FileNotFoundError(args.image)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    elif default_ref_image.exists():
        print(f"Using reference image: {_rel(default_ref_image)}")
        image_bgr = cv2.imread(str(default_ref_image))
        if image_bgr is None:
            raise RuntimeError(f"Cannot read reference image: {default_ref_image}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        print("No --image and no reference image found, capturing live frame...")
        image_rgb = capture_rgb_frame(config)

    # --- resolve quad ---
    quad_px = scan_quad_px
    if args.auto_quad and args.manual_quad:
        raise ValueError("Use only one of --auto-quad or --manual-quad.")

    if quad_px is None and not args.manual_quad and not args.auto_quad:
        # try to reuse quad from layout JSON (saved by mark_keyboard_layout.py)
        layout_data = load_json(args.layout)
        if "quad_px_used" in layout_data:
            quad_px = np.array(layout_data["quad_px_used"], dtype=np.float32)
            print(f"Reusing quad from layout: {_rel(args.layout)}")

    targets = build_key_targets(
        image_rgb=image_rgb,
        layout_path=args.layout,
        homography_path=args.homography,
        press_config_path=args.press_config,
        quad_px=quad_px,
        manual_quad=args.manual_quad,
        layout_orientation=args.layout_orientation,
        target_source=args.target_source,
    )
    print(f"built {len(targets['keys'])} key targets")
    print(f"saved {root / 'runtime' / 'current_key_targets_3d.json'}")
    print(f"saved {root / 'runtime' / 'debug_overlay.png'}")


if __name__ == "__main__":
    main()
