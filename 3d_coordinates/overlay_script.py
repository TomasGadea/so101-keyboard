"""
Draw keyboard letter and corner overlays on an image.

How to run as a script:
    python overlay_script.py
    python overlay_script.py path/to/frame.png --out my_overlay.png
    python overlay_script.py frame.png --keys keys.json --corners corners.json --out overlay.png

Programmatic use:
    from overlay_script import run
    run(image_path, keys_data, corners_data=None, output_path="overlay.png")
"""

import argparse
import json
from pathlib import Path

import cv2

DEFAULT_IMAGE_PATH = 'outputs/captured_images/opencv__dev_video4.png'
TARGET_SIZE = (640, 480)

VERBOSE = False


def vprint(*args, **kwargs):
    """print() that only emits when this module's VERBOSE flag is set."""
    if VERBOSE:
        print(*args, **kwargs)


def coord_scale(coord_system: str):
    """Return divisor to bring x/y into [0, 1], or None to treat as pixels."""
    if coord_system.startswith('normalized_'):
        return float(coord_system.split('_', 1)[1])
    if coord_system == 'normalized':
        return 1.0
    return None


def to_pixel_xy(x: float, y: float, scale, width: int, height: int):
    if scale is not None:
        return (int(round(x / scale * width)),
                int(round(y / scale * height)))
    return int(round(x)), int(round(y))


def run(image_path, keys_data, corners_data=None, output_path="keyboard_overlay.png",
        verbose=False):
    """Draw key letters (and optional corners) on the image and save it.

    Args:
        image_path: path to the input image.
        keys_data: dict with schema
            {"letters": [{"char": "Q", "x": .., "y": ..}, ...],
             "coordinate_system": "normalized_1000"}.
        corners_data: optional dict with schema
            {"coordinates": {"top_left": {"x": .., "y": ..}, ...},
             "coordinate_system": "normalized_1000"}.
        output_path: where to write the annotated image.

    Returns:
        Path to the written overlay image.
    """
    global VERBOSE
    VERBOSE = verbose

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    if (image.shape[1], image.shape[0]) != TARGET_SIZE:
        vprint(f"Resizing image from {(image.shape[1], image.shape[0])} to {TARGET_SIZE}.")
        image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]

    keys_scale = coord_scale(keys_data['coordinate_system'])
    for item in keys_data['letters']:
        char = str(item['char']).upper()
        px, py = to_pixel_xy(float(item['x']), float(item['y']),
                             keys_scale, width, height)
        cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
        cv2.putText(image, char, (px - 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if corners_data is not None:
        corner_keys = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        coords = corners_data['coordinates']
        corners_scale = coord_scale(corners_data['coordinate_system'])

        corner_pts = []
        for name in corner_keys:
            if name not in coords:
                continue
            entry = coords[name]
            px, py = to_pixel_xy(float(entry['x']), float(entry['y']),
                                 corners_scale, width, height)
            corner_pts.append((px, py))
            cv2.circle(image, (px, py), 7, (0, 255, 0), 2)
            cv2.putText(image, name, (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if len(corner_pts) >= 2:
            for i in range(len(corner_pts)):
                cv2.line(image, corner_pts[i],
                         corner_pts[(i + 1) % len(corner_pts)],
                         (0, 255, 0), 2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    vprint(f"Overlay image saved as {output_path}")
    return output_path


def _main():
    parser = argparse.ArgumentParser(
        description="Overlay keyboard letters and corners on an image.")
    parser.add_argument('image', nargs='?', default=DEFAULT_IMAGE_PATH)
    parser.add_argument('--keys', default='keys.json')
    parser.add_argument('--corners', default='corners.json')
    parser.add_argument('--out', default='keyboard_overlay.png')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress information.')
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    with open(args.keys) as f:
        keys_data = json.load(f)

    corners_data = None
    try:
        with open(args.corners) as f:
            corners_data = json.load(f)
    except FileNotFoundError:
        vprint(f"No corners file at {args.corners}; skipping corner overlay.")

    run(args.image, keys_data, corners_data, args.out, verbose=args.verbose)


if __name__ == "__main__":
    _main()
