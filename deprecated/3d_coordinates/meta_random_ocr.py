"""Run ocr_keyboard_coords.py on 100 random images with a random letter in --strong mode."""

import random
import shutil
import string
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW_APPROACH = Path("/home/till/Documents/robot_learning/robot_learning_project/new_approach_with_homography")
IMAGES_DIR = NEW_APPROACH / "images"
SCRIPT = NEW_APPROACH / "ocr_keyboard_coords.py"
INPUT_OUT_DIR = HERE / "a"
OVERLAY_OUT_DIR = HERE / "b"
N = 100


def main():
    images = [p for p in IMAGES_DIR.iterdir()
              if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not images:
        sys.exit(f"No images found in {IMAGES_DIR}")

    INPUT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(1, N + 1):
        img = random.choice(images)
        key = random.choice(string.ascii_uppercase)
        input_copy = INPUT_OUT_DIR / f"{i}.jpg"
        overlay_out = OVERLAY_OUT_DIR / f"{i}.jpg"

        shutil.copy2(img, input_copy)
        print(f"\n=== [{i}/{N}] image={img.name} key={key} ===")
        subprocess.run(
            [sys.executable, str(SCRIPT), str(img), key,
             "--overlay-out", str(overlay_out), "--strong"],
            check=False,
        )


if __name__ == "__main__":
    main()
