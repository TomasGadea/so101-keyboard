"""Run pipeline.py on 10 random images with a random letter for --key."""

import random
import shutil
import string
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES_DIR = Path("/home/till/Documents/robot_learning/robot_learning_project/new_approach_with_homography/images")
PIPELINE = HERE / "pipeline.py"
INPUT_OUT_DIR = HERE / "a"
OVERLAY_OUT_DIR = HERE / "b"
N = 10


def main():
    images = [p for p in IMAGES_DIR.iterdir()
              if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if len(images) < N:
        sys.exit(f"Need at least {N} images in {IMAGES_DIR}, found {len(images)}")

    INPUT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_OUT_DIR.mkdir(parents=True, exist_ok=True)

    picks = random.sample(images, N)
    for i, img in enumerate(picks, 1):
        key = random.choice(string.ascii_uppercase)
        input_copy = INPUT_OUT_DIR / f"{i}.jpg"
        overlay_out = OVERLAY_OUT_DIR / f"{i}.jpg"

        shutil.copy2(img, input_copy)
        print(f"\n=== [{i}/{N}] image={img.name} key={key} ===")
        subprocess.run(
            [sys.executable, str(PIPELINE),
             "--image", str(img),
             "--key", key,
             "--overlay-out", str(overlay_out)],
            check=False,
        )


if __name__ == "__main__":
    main()
