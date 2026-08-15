"""
Compute 3D coordinates of keyboard letters.

How to run:
    # Reads corners.json + keys.json from the current directory and writes
    # keys_3d.json with each letter's (x, y, z) on the keyboard plane:
    python letters_to_3d.py

Edit the CORNERS_3D / CAMERA_3D constants below to match your physical setup.

Inputs:
    - 3D coordinates of the 4 laptop corners (TL, TR, BR, BL).
    - 3D coordinates of the camera (used for sanity checks; not required for
      the core computation).
    - corners.json with image-space coords of the 4 corners.
    - keys.json with image-space coords of the letters.

Method:
    Letters are assumed to lie on the plane spanned by the laptop corners. We
    parameterize that plane with two non-orthogonal basis vectors derived from
    the corners, compute (s, t) plane coordinates for each corner, then fit a
    planar homography H: (u, v) -> (s, t) from the 4 corner correspondences.
    Each letter's image (u, v) is mapped through H to (s, t), then lifted back
    to 3D as origin + s * e1 + t * e2.
"""

import json

import cv2
import numpy as np

# ---------------- Inputs ----------------
# 3D corners, in order: top-left, top-right, bottom-right, bottom-left.
# x [m]: bottom edge sits at the far side of an A4 sheet (0.210 m); top edge
#        is keyboard depth (0.132 m) further away -> 0.342 m.
FIRST_TO_SECOND_MOTOR_DISPLACEMENT = 0.04
A4_WIDTH = 0.210    # short edge of A4 paper [m]
KEYBOARD_DEPTH = 0.132  # front-to-back size of the keyboard [m]
X_BOTTOM = A4_WIDTH + FIRST_TO_SECOND_MOTOR_DISPLACEMENT
X_TOP = A4_WIDTH + KEYBOARD_DEPTH

Y_DISPLACEMENT = 0.138
KEYBOARD_WIDTH = 0.279

COORD_CENTER_HEIGHT = 0.052 # 0.123
KEYBOARD_HEIGHT = 0.004 # TO 0.011 

CORNERS_3D = np.array([
    [X_TOP,    KEYBOARD_WIDTH - Y_DISPLACEMENT, - COORD_CENTER_HEIGHT + KEYBOARD_HEIGHT],   # TL
    [X_TOP,    - Y_DISPLACEMENT, - COORD_CENTER_HEIGHT + KEYBOARD_HEIGHT],   # TR  (y/z placeholders — fill in real values)
    [X_BOTTOM, - Y_DISPLACEMENT, - COORD_CENTER_HEIGHT + KEYBOARD_HEIGHT],   # BR
    [X_BOTTOM, KEYBOARD_WIDTH - Y_DISPLACEMENT, - COORD_CENTER_HEIGHT + KEYBOARD_HEIGHT],   # BL
], dtype=np.float64)

# Camera 3D position (used only for an optional sanity check).
# TODO: Fill in the actual camera position.
CAMERA_3D = np.array([0.17, 0.20, 0.50], dtype=np.float64)

CORNERS_JSON = "corners.json"
KEYS_JSON = "keys.json"
OUTPUT_JSON = "keys_3d.json"

CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]


def coord_scale(coord_system: str) -> float:
    """Return divisor to bring x/y into [0, 1]."""
    if coord_system.startswith("normalized_"):
        return float(coord_system.split("_", 1)[1])
    if coord_system == "normalized":
        return 1.0
    raise ValueError(f"Unsupported coordinate_system: {coord_system!r}")


def load_corners_uv(path: str) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    scale = coord_scale(data["coordinate_system"])
    coords = data["coordinates"]
    return np.array(
        [[coords[name]["x"] / scale, coords[name]["y"] / scale]
         for name in CORNER_ORDER],
        dtype=np.float64,
    )


def load_letters_uv(path: str):
    with open(path) as f:
        data = json.load(f)
    scale = coord_scale(data["coordinate_system"])
    chars = [item["char"] for item in data["letters"]]
    uv = np.array(
        [[item["x"] / scale, item["y"] / scale] for item in data["letters"]],
        dtype=np.float64,
    )
    return chars, uv


def plane_basis(corners_3d: np.ndarray):
    """Return (origin, e1, e2, normal) defining the laptop plane."""
    origin = corners_3d[0]
    e1 = corners_3d[1] - origin   # along the top edge (TL -> TR)
    e2 = corners_3d[3] - origin   # along the left edge (TL -> BL)
    normal = np.cross(e1, e2)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-12:
        raise ValueError("Corners are colinear; cannot define a plane.")
    normal = normal / n_norm
    return origin, e1, e2, normal


def corners_in_plane_coords(corners_3d, origin, e1, e2, normal):
    """Express each 3D corner as (s, t) in the (e1, e2) basis on the plane."""
    basis = np.column_stack([e1, e2, normal])
    basis_inv = np.linalg.inv(basis)
    rel = corners_3d - origin                   # (4, 3)
    coords = (basis_inv @ rel.T).T              # (4, 3) -> last col should be ~0
    if np.max(np.abs(coords[:, 2])) > 1e-6:
        raise ValueError("Corners are not coplanar within tolerance.")
    return coords[:, :2]


def main():
    origin, e1, e2, normal = plane_basis(CORNERS_3D)
    corners_st = corners_in_plane_coords(CORNERS_3D, origin, e1, e2, normal)

    corners_uv = load_corners_uv(CORNERS_JSON)
    chars, letters_uv = load_letters_uv(KEYS_JSON)

    # Homography from image [0, 1] coords to plane (s, t) coords.
    H = cv2.getPerspectiveTransform(
        corners_uv.astype(np.float32),
        corners_st.astype(np.float32),
    )

    # Optional sanity check: camera should not lie on the plane.
    signed_dist = np.dot(CAMERA_3D - origin, normal)
    print(f"Camera signed distance to laptop plane: {signed_dist:+.4f}")

    uv = letters_uv.astype(np.float32).reshape(-1, 1, 2)
    st = cv2.perspectiveTransform(uv, H).reshape(-1, 2)
    points_3d = origin + st[:, 0:1] * e1 + st[:, 1:2] * e2

    output = {
        "letters": [
            {"char": c,
             "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
            for c, p in zip(chars, points_3d)
        ]
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(chars)} letters with 3D coords to {OUTPUT_JSON}")
    for c, p in zip(chars, points_3d):
        print(f"  {c}: ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")


if __name__ == "__main__":
    main()
