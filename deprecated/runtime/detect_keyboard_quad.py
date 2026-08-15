#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.common import load_json, repo_root, save_json
from runtime.geometry import order_quad_points


def _maybe_rgb_to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _quad_score(
    quad: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    min_aspect_ratio: float = 1.8,
    max_area_ratio: float = 0.75,
) -> tuple[float, dict] | None:
    h, w = image_shape[:2]
    ordered = order_quad_points(quad)
    area = abs(cv2.contourArea(ordered.astype(np.float32)))
    area_ratio = area / float(w * h)
    if not 0.10 <= area_ratio <= max_area_ratio:
        return None

    side_lengths = [
        np.linalg.norm(ordered[(i + 1) % 4] - ordered[i])
        for i in range(4)
    ]
    short = max(min(side_lengths), 1e-6)
    long = max(side_lengths)
    aspect_ratio = long / short
    if not min_aspect_ratio <= aspect_ratio <= 8.0:
        return None

    rect = cv2.minAreaRect(ordered.astype(np.float32))
    angle = rect[2]
    rw, rh = rect[1]
    if rw < rh:
        angle += 90.0
    angle = ((angle + 90.0) % 180.0) - 90.0
    if abs(angle) > 45.0:
        return None

    center = ordered.mean(axis=0)
    center_y_norm = center[1] / max(h, 1)
    center_score = 1.0 - min(abs(center_y_norm - 0.62) / 0.62, 1.0)
    aspect_score = 1.0 - min(abs(aspect_ratio - 3.2) / 4.8, 1.0)
    skew = abs(side_lengths[0] - side_lengths[2]) / max(side_lengths[0], side_lengths[2], 1e-6)
    skew += abs(side_lengths[1] - side_lengths[3]) / max(side_lengths[1], side_lengths[3], 1e-6)
    skew_score = max(0.0, 1.0 - 0.5 * skew)
    area_score = 1.0 - min(abs(area_ratio - 0.35) / 0.35, 1.0)
    score = 0.35 * area_score + 0.25 * aspect_score + 0.25 * center_score + 0.15 * skew_score
    return float(max(0.0, min(score, 1.0))), {
        "area_ratio": float(area_ratio),
        "aspect_ratio": float(aspect_ratio),
        "angle_deg": float(angle),
    }


def _detect_keyboard_body_quad(image_rgb: np.ndarray) -> tuple[np.ndarray, float, dict] | None:
    """
    Detect the full keyboard body as a large dark object.

    This is intentionally tried before generic edge quadrilateral detection.
    Generic contours often lock onto a tidy sub-rectangle of keys; the keyboard
    target frame needs the outer body instead.
    """
    gray = _maybe_rgb_to_gray(image_rgb)
    h, w = gray.shape[:2]

    best: tuple[float, np.ndarray, dict] | None = None
    for threshold in (70, 90, 110, 130):
        mask = (gray < threshold).astype(np.uint8) * 255
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_area_ratio = cv2.contourArea(contour) / float(w * h)
            if not 0.08 <= contour_area_ratio <= 0.80:
                continue

            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if min(rw, rh) < 80:
                continue

            aspect_ratio = max(rw, rh) / max(min(rw, rh), 1e-6)
            if not 1.35 <= aspect_ratio <= 6.5:
                continue

            quad = cv2.boxPoints(rect).astype(np.float32)
            ordered_quad = order_quad_points(quad)
            # A common false positive is "the bottom part of the image" when
            # the keyboard is cropped by the frame. That is not enough to infer
            # a reliable keyboard coordinate system.
            bottom_edge_vertices = sum(y >= h - 4 for _, y in ordered_quad)
            visible_y = np.clip(ordered_quad[:, 1], 0, h - 1)
            visible_height = float(visible_y.max() - visible_y.min())
            if bottom_edge_vertices >= 2 and visible_height < 0.48 * h:
                continue

            border_vertices = sum(
                x <= 2 or x >= w - 3 or y <= 2 or y >= h - 3
                for x, y in ordered_quad
            )
            min_x, min_y = ordered_quad.min(axis=0)
            max_x, max_y = ordered_quad.max(axis=0)
            if border_vertices >= 4 and 0 <= min_x and max_x <= w - 1 and 0 <= min_y and max_y <= h - 1:
                continue

            scored = _quad_score(
                quad,
                image_rgb.shape,
                min_aspect_ratio=1.35,
                max_area_ratio=0.95,
            )
            if scored is None:
                continue

            base_score, details = scored
            # Prefer the broad dark keyboard body over smaller key clusters.
            area_score = min(contour_area_ratio / 0.45, 1.0)
            aspect_score = 1.0 - min(abs(aspect_ratio - 3.0) / 3.0, 1.0)
            angle_abs = abs(details["angle_deg"])
            border_penalty = 0.12 * border_vertices
            if border_vertices >= 3 and angle_abs < 5.0:
                border_penalty += 0.12
            score = 0.35 * base_score + 0.45 * area_score + 0.20 * aspect_score - border_penalty
            details.update(
                {
                    "detector": "dark_body",
                    "threshold": int(threshold),
                    "contour_area_ratio": float(contour_area_ratio),
                    "border_vertices": int(border_vertices),
                }
            )

            if best is None or score > best[0]:
                best = (float(max(0.0, score)), ordered_quad, details)

    return best


def detect_keyboard_quad_cv(image_rgb: np.ndarray) -> tuple[np.ndarray, float, dict]:
    body_best = _detect_keyboard_body_quad(image_rgb)
    if body_best is not None:
        return body_best[1], body_best[0], body_best[2]

    gray = _maybe_rgb_to_gray(image_rgb)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: tuple[float, np.ndarray, dict] | None = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        for eps_scale in (0.015, 0.02, 0.03, 0.04, 0.06):
            approx = cv2.approxPolyDP(contour, eps_scale * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = approx.reshape(4, 2).astype(np.float32)
            scored = _quad_score(quad, image_rgb.shape)
            if scored is None:
                continue
            score, details = scored
            if best is None or score > best[0]:
                best = (score, order_quad_points(quad), details)

    if best is None:
        raise RuntimeError("Could not detect a plausible keyboard quadrilateral.")
    return best[1], best[0], best[2]


def manual_select_quad(image_rgb: np.ndarray) -> np.ndarray:
    """Open a matplotlib window; user clicks the 4 corners in order.

    Order: top-left, top-right, bottom-right, bottom-left. cv2.imshow won't
    work in WSL with opencv-python-headless installed, so we use matplotlib
    (TkAgg) which we know works through WSLg.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
    points: list[list[float]] = []

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(image_rgb)
    ax.set_title(f"Click {labels[0]} corner of the keyboard")
    ax.set_xlabel("close window when 4 points placed; press 'r' to reset, 'esc' to abort")
    line, = ax.plot([], [], "-", color="#ffd400", lw=2)
    pts_marker, = ax.plot([], [], "o", color="red", markersize=8)
    aborted = {"v": False}

    def redraw():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        pts_marker.set_data(xs, ys)
        if len(points) == 4:
            line.set_data(xs + [xs[0]], ys + [ys[0]])
            ax.set_title("4 points placed — close window to accept (or 'r' to reset)")
        else:
            line.set_data([], [])
            ax.set_title(f"Click {labels[len(points)]} corner of the keyboard")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.xdata is None or event.ydata is None or event.button != 1:
            return
        if len(points) >= 4:
            return
        u, v = float(event.xdata), float(event.ydata)
        points.append([u, v])
        print(f"clicked {labels[len(points) - 1]}: [{u:.0f}, {v:.0f}]")
        redraw()

    def on_key(event):
        if event.key == "r":
            points.clear()
            print("reset")
            redraw()
        elif event.key in ("escape",):
            aborted["v"] = True
            plt.close(fig)
        elif event.key in ("enter", " "):
            if len(points) == 4:
                plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if aborted["v"]:
        raise KeyboardInterrupt("Manual quad selection aborted.")
    if len(points) != 4:
        raise RuntimeError(f"Need 4 corners, got {len(points)}")
    return np.asarray(points, dtype=np.float32)


def save_quad_json(path: Path, quad_px: np.ndarray, confidence: float, method: str, image_shape: tuple[int, ...]) -> dict:
    h, w = image_shape[:2]
    quad = [[int(round(float(x))), int(round(float(y)))] for x, y in order_quad_points(quad_px)]
    data = {
        "quad_px": quad,
        "confidence": float(confidence),
        "method": method,
        "image_width": int(w),
        "image_height": int(h),
    }
    save_json(path, data)
    return data


def _draw_quad_debug(image_rgb: np.ndarray, quad_px: np.ndarray, out_path: Path) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    quad = order_quad_points(quad_px).astype(np.int32)
    cv2.polylines(image_bgr, [quad], True, (0, 220, 255), 3)
    for i, (x, y) in enumerate(quad):
        cv2.circle(image_bgr, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(image_bgr, str(i), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image_bgr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect current keyboard quadrilateral.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=str(repo_root() / "runtime" / "quad_debug.png"))
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--from-json", dest="from_json", default=None)
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if args.from_json:
        data = load_json(args.from_json)
        quad = np.asarray(data["quad_px"], dtype=np.float32)
        confidence = float(data.get("confidence", 1.0))
        method = data.get("method", "from_json")
    elif args.manual:
        quad = manual_select_quad(image_rgb)
        confidence = 1.0
        method = "manual"
    else:
        quad, confidence, _ = detect_keyboard_quad_cv(image_rgb)
        method = "cv_contour"

    json_path = repo_root() / "runtime" / "current_keyboard_quad.json"
    data = save_quad_json(json_path, quad, confidence, method, image_rgb.shape)
    _draw_quad_debug(image_rgb, np.asarray(data["quad_px"], dtype=np.float32), Path(args.out))
    print(f"saved {json_path}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
