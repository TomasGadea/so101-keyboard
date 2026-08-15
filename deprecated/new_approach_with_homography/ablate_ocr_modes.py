"""Compare standard OCR and strong OCR on saved keyboard frames.

This is a no-robot/no-camera ablation. It reports speed, fit quality, success
rate, and coordinate deltas between the two OCR modes. Synthetic in-canvas
rotations can be added to stress test keyboard tilt.

Example:
    python new_approach_with_homography/ablate_ocr_modes.py \
        --limit 5 --rotations 0 25 -25 --keys H ENTER SPACE R L
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "new_approach_with_homography"))

import ocr_keyboard_coords  # noqa: E402


DEFAULT_KEYS = ["H", "ENTER", "SPACE", "R", "L"]
IMAGE_SIZE = (640, 480)


def select_images(image_args, image_dir, limit):
    if image_args:
        return [Path(p) for p in image_args]
    images = [
        p for p in sorted(Path(image_dir).glob("*.jpg"),
                          key=lambda p: p.stat().st_mtime,
                          reverse=True)
        if p.stat().st_size > 30000
    ]
    return images[:limit]


def rotate_in_canvas(image, angle):
    if abs(angle) < 1e-6:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def output_for(keys, fit, width, height):
    data = ocr_keyboard_coords._build_output(keys, fit.projected, width, height)
    return {
        entry["char"]: (float(entry["x"]), float(entry["y"]))
        for entry in data["letters"]
    }


def run_mode(mode, image, keys, strong_deadline_s=None):
    height, width = image.shape[:2]
    t0 = time.perf_counter()
    log_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_buf):
            if mode == "standard":
                detections, fit = ocr_keyboard_coords._locate_standard(
                    image, width=width, height=height, keys=keys
                )
            elif mode == "strong":
                detections, fit = ocr_keyboard_coords._locate_strong(
                    image, width=width, height=height, keys=keys,
                    deadline_s=strong_deadline_s
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")
        dt = time.perf_counter() - t0
        return {
            "ok": True,
            "dt_s": dt,
            "detected_unique": len({det.label for det in detections}),
            "detected_labels": sorted({det.label for det in detections}),
            "inliers": len(fit.inlier_labels),
            "inlier_rows": len(fit.inlier_rows),
            "mean_error_px": fit.mean_error_px,
            "coords": output_for(keys, fit, width, height),
            "log": log_buf.getvalue().splitlines(),
        }
    except Exception as exc:
        dt = time.perf_counter() - t0
        return {
            "ok": False,
            "dt_s": dt,
            "error": f"{type(exc).__name__}: {exc}",
            "log": log_buf.getvalue().splitlines(),
        }


def coord_delta_px(keys, a, b):
    if not (a.get("ok") and b.get("ok")):
        return None
    width, height = IMAGE_SIZE
    dists = []
    per_key = {}
    for key in keys:
        ax, ay = a["coords"][key]
        bx, by = b["coords"][key]
        dist = math.hypot((ax - bx) / 1000.0 * width,
                          (ay - by) / 1000.0 * height)
        dists.append(dist)
        per_key[key] = dist
    return {
        "avg_px": statistics.mean(dists),
        "max_px": max(dists),
        "per_key_px": per_key,
    }


def summarize(rows, mode):
    entries = [row[mode] for row in rows]
    ok = [entry for entry in entries if entry["ok"]]
    out = {
        "n": len(entries),
        "ok": len(ok),
        "fail": len(entries) - len(ok),
        "success_rate": len(ok) / len(entries) if entries else 0.0,
    }
    if ok:
        out.update({
            "mean_s": statistics.mean(entry["dt_s"] for entry in ok),
            "median_s": statistics.median(entry["dt_s"] for entry in ok),
            "mean_inliers": statistics.mean(entry["inliers"] for entry in ok),
            "median_inliers": statistics.median(entry["inliers"] for entry in ok),
            "mean_rows": statistics.mean(entry["inlier_rows"] for entry in ok),
            "mean_error_px": statistics.mean(entry["mean_error_px"] for entry in ok),
        })
    return out


def summarize_deltas(rows):
    deltas = [row["delta"] for row in rows if row.get("delta")]
    if not deltas:
        return {}
    return {
        "n": len(deltas),
        "avg_px_mean": statistics.mean(delta["avg_px"] for delta in deltas),
        "avg_px_median": statistics.median(delta["avg_px"] for delta in deltas),
        "max_px_max": max(delta["max_px"] for delta in deltas),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", help="Specific image files to compare.")
    parser.add_argument("--image-dir", default=str(PROJECT / "new_approach_with_homography" / "images"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    parser.add_argument("--rotations", nargs="+", type=float, default=[0.0])
    parser.add_argument("--strong-deadline-s", type=float, default=None,
                        help="Apply a deadline to strong OCR localization.")
    parser.add_argument("--out-dir", default="/tmp/ocr_modes_ablation")
    args = parser.parse_args()

    keys = [key.upper() for key in args.keys]
    images = select_images(args.images, args.image_dir, args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in images:
        base = cv2.imread(str(image_path))
        if base is None:
            print(f"[ablate] skip unreadable image: {image_path}")
            continue
        if (base.shape[1], base.shape[0]) != IMAGE_SIZE:
            base = cv2.resize(base, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        for angle in args.rotations:
            image = rotate_in_canvas(base, angle)
            print(f"\n=== {image_path.name} angle={angle:+.0f} ===")
            standard = run_mode("standard", image, keys)
            strong = run_mode("strong", image, keys,
                              strong_deadline_s=args.strong_deadline_s)
            delta = coord_delta_px(keys, standard, strong)
            row = {
                "image": image_path.name,
                "angle": angle,
                "standard": standard,
                "strong": strong,
                "delta": delta,
            }
            rows.append(row)

            for name, result in (("standard", standard), ("strong", strong)):
                if result["ok"]:
                    print(
                        f"{name:8s} ok  {result['dt_s']:.2f}s  "
                        f"inliers={result['inliers']:2d} "
                        f"rows={result['inlier_rows']} "
                        f"err={result['mean_error_px']:.1f}px"
                    )
                else:
                    print(f"{name:8s} FAIL {result['dt_s']:.2f}s  {result['error']}")
            if delta:
                print(
                    f"delta    avg={delta['avg_px']:.1f}px "
                    f"max={delta['max_px']:.1f}px"
                )

    report = {
        "keys": keys,
        "rotations": args.rotations,
        "strong_deadline_s": args.strong_deadline_s,
        "rows": rows,
        "summary": {
            "standard": summarize(rows, "standard"),
            "strong": summarize(rows, "strong"),
            "delta": summarize_deltas(rows),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"\n[ablate] wrote {out_dir / 'summary.json'}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
