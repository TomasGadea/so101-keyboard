from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def draw_key_overlay(
    image_rgb: np.ndarray,
    quad_px: np.ndarray,
    key_pixels: dict[str, np.ndarray],
    out_path: str | Path,
    show_all: bool = True,
) -> None:
    """
    Draw keyboard quadrilateral, key centers, labels, and emphasized anchors.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    quad = np.asarray(quad_px, dtype=np.int32).reshape(4, 2)
    cv2.polylines(image_bgr, [quad], isClosed=True, color=(0, 220, 255), thickness=3)

    emphasized = {"Q", "A", "M", "SPACE", "ENTER"}
    for key, point in key_pixels.items():
        if not show_all and key not in emphasized:
            continue
        x, y = np.asarray(point, dtype=float)
        center = (int(round(x)), int(round(y)))
        is_emph = key in emphasized
        color = (0, 0, 255) if is_emph else (0, 180, 0)
        radius = 6 if is_emph else 3
        thickness = -1 if is_emph else 1
        cv2.circle(image_bgr, center, radius, color, thickness)
        cv2.putText(
            image_bgr,
            key,
            (center[0] + 6, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 if len(key) <= 1 else 0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image_bgr)
