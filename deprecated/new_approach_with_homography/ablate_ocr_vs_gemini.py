"""Compare local OCR coordinates against a Gemini model on saved frames.

No robot or camera is used. The script writes one OCR JSON, one Gemini JSON,
and a summary.json under the chosen output directory.

Example:
    python new_approach_with_homography/ablate_ocr_vs_gemini.py \
        --limit 5 --keys SPACE ENTER R L H
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "new_approach_with_homography"))
sys.path.insert(0, str(PROJECT / "3d_coordinates"))

import ocr_keyboard_coords  # noqa: E402
import vlm_keyboard_coords  # noqa: E402


DEFAULT_MODEL = "models/gemini-3.1-pro-preview"
DEFAULT_KEYS = ["SPACE", "ENTER", "R", "L", "H"]
IMAGE_SIZE = (640, 480)


def build_requested_keys_prompt(keys):
    keys_list = ", ".join(f'"{key}"' for key in keys)
    entries = ",\n    ".join(
        f'{{"char": "{key}", "x": number, "y": number}}' for key in keys
    )
    return f"""Analyze the image. Your task is to detect and provide the normalized coordinates for the center of each of the following keyboard keys: {keys_list}.
Use a normalized coordinate system where [0, 0] is the top-left and [1000, 1000] is the bottom-right of the image.
For letter keys, return the center of the physical keycap. Use "SPACE" for the space bar and "ENTER" for the Enter / Return key.
The ENTER key is on the right side of the keyboard and may be a wide rectangular key or a tall L-shaped key depending on the layout; return the center of the full Enter key.
Return the data strictly as a JSON object in the following format:

{{
  "letters": [
    {entries}
  ],
  "count": {len(keys)},
  "coordinate_system": "normalized_1000"
}}
Do not provide any conversational text, only the JSON block."""


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


def by_char(data):
    return {
        entry["char"].upper(): (float(entry["x"]), float(entry["y"]))
        for entry in data["letters"]
    }


def compare_keys(keys, ocr_data, gemini_data):
    ocr_by_char = by_char(ocr_data)
    gemini_by_char = by_char(gemini_data)
    rows = []
    width, height = IMAGE_SIZE
    for key in keys:
        if key not in ocr_by_char or key not in gemini_by_char:
            rows.append({
                "key": key,
                "missing": True,
                "ocr_present": key in ocr_by_char,
                "gemini_present": key in gemini_by_char,
            })
            continue
        ox, oy = ocr_by_char[key]
        gx, gy = gemini_by_char[key]
        dx = ox - gx
        dy = oy - gy
        rows.append({
            "key": key,
            "ocr": [ox, oy],
            "gemini": [gx, gy],
            "dx_norm": dx,
            "dy_norm": dy,
            "dist_norm": math.hypot(dx, dy),
            "dist_px": math.hypot(dx / 1000.0 * width, dy / 1000.0 * height),
        })
    return rows


def summarize(rows):
    valid = [r for r in rows if "error" not in r]
    if not valid:
        return {}
    return {
        "n": len(valid),
        "ocr_mean_s": statistics.mean(r["ocr_dt"] for r in valid),
        "ocr_median_s": statistics.median(r["ocr_dt"] for r in valid),
        "gemini_mean_s": statistics.mean(r["gemini_dt"] for r in valid),
        "gemini_median_s": statistics.median(r["gemini_dt"] for r in valid),
        "avg_delta_px_mean": statistics.mean(r["avg_dist_px"] for r in valid),
        "avg_delta_px_median": statistics.median(r["avg_dist_px"] for r in valid),
        "max_delta_px_max": max(r["max_dist_px"] for r in valid),
    }


def main():
    load_dotenv(PROJECT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", help="Specific image files to compare.")
    parser.add_argument("--image-dir", default=str(PROJECT / "new_approach_with_homography" / "images"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default="/tmp/ocr_vs_gemini_pro_ablation")
    args = parser.parse_args()

    keys = [key.upper() for key in args.keys]
    images = select_images(args.images, args.image_dir, args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    prompt = build_requested_keys_prompt(keys)
    for idx, image_path in enumerate(images):
        print(f"\n=== [{idx}] {image_path.name} ===")
        try:
            t0 = time.perf_counter()
            ocr_data = ocr_keyboard_coords.run(
                image_path=image_path,
                keys=keys,
                keys_json=out_dir / f"{idx:02d}_ocr.json",
            )
            ocr_dt = time.perf_counter() - t0
            print(f"[ablate] OCR total: {ocr_dt:.2f}s")
        except Exception as exc:
            rows.append({"image": image_path.name,
                         "error": f"OCR {type(exc).__name__}: {exc}"})
            print(f"[ablate] OCR failed: {exc}")
            continue

        try:
            t0 = time.perf_counter()
            gemini_data, _ = vlm_keyboard_coords.run(
                image_path=str(image_path),
                keys_json=str(out_dir / f"{idx:02d}_gemini.json"),
                letters_prompt=prompt,
                model=args.model,
            )
            gemini_dt = time.perf_counter() - t0
            print(f"[ablate] Gemini total: {gemini_dt:.2f}s")
        except Exception as exc:
            rows.append({
                "image": image_path.name,
                "ocr_dt": ocr_dt,
                "error": f"Gemini {type(exc).__name__}: {exc}",
            })
            print(f"[ablate] Gemini failed: {exc}")
            continue

        key_rows = compare_keys(keys, ocr_data, gemini_data)
        present = [r for r in key_rows if not r.get("missing")]
        avg_px = statistics.mean(r["dist_px"] for r in present)
        max_px = max(r["dist_px"] for r in present)
        row = {
            "image": image_path.name,
            "ocr_dt": ocr_dt,
            "gemini_dt": gemini_dt,
            "avg_dist_px": avg_px,
            "max_dist_px": max_px,
            "keys": key_rows,
        }
        rows.append(row)
        print(f"[ablate] avg_delta_px={avg_px:.1f} max_delta_px={max_px:.1f}")
        for key_row in key_rows:
            if key_row.get("missing"):
                print(f"  {key_row['key']}: missing")
            else:
                print(f"  {key_row['key']}: {key_row['dist_px']:.1f}px")

    report = {
        "model": args.model,
        "keys": keys,
        "rows": rows,
        "summary": summarize(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"\n[ablate] wrote {out_dir / 'summary.json'}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
