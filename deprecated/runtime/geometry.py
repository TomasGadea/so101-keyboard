from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from runtime.common import load_json


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Input: 4 unordered points, shape [4,2].
    Output: top_left, top_right, bottom_right, bottom_left.
    """
    arr = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - center[1], arr[:, 0] - center[0])
    ordered = arr[np.argsort(angles)]
    tl = ordered[np.argmin(ordered.sum(axis=1))]
    br = ordered[np.argmax(ordered.sum(axis=1))]
    remaining = [p for p in ordered if not np.allclose(p, tl) and not np.allclose(p, br)]
    if remaining[0][0] > remaining[1][0]:
        tr, bl = remaining[0], remaining[1]
    else:
        tr, bl = remaining[1], remaining[0]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def apply_homography(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    mat = np.asarray(H, dtype=np.float64)
    return cv2.perspectiveTransform(points, mat).reshape(-1, 2)


def keyboard_unit_to_image_homography(quad_px: np.ndarray) -> np.ndarray:
    quad = order_quad_points(quad_px).astype(np.float32)
    unit = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(unit, quad)


def image_to_base_homography_from_file(path: str | Path) -> np.ndarray:
    data = load_json(path)
    if "H_image_to_base_xy" not in data:
        raise KeyError(f"{path} does not contain H_image_to_base_xy")
    return np.asarray(data["H_image_to_base_xy"], dtype=np.float64)


def normalized_keys_to_image(layout: dict, quad_px: np.ndarray) -> dict[str, np.ndarray]:
    H = keyboard_unit_to_image_homography(quad_px)
    keys = layout.get("keys")
    if not isinstance(keys, dict):
        raise KeyError('layout must contain a "keys" object')
    names = list(keys.keys())
    uv = np.array([[keys[name]["u"], keys[name]["v"]] for name in names], dtype=np.float32)
    pixels = apply_homography(uv, H)
    return {name: pixels[i].astype(np.float64) for i, name in enumerate(names)}


def image_keys_to_base_xy(
    key_px: dict[str, np.ndarray],
    H_image_to_base: np.ndarray,
) -> dict[str, np.ndarray]:
    names = list(key_px.keys())
    pixels = np.array([key_px[name] for name in names], dtype=np.float32)
    xy = apply_homography(pixels, H_image_to_base)
    return {name: xy[i].astype(np.float64) for i, name in enumerate(names)}
