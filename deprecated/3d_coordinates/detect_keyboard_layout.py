"""
Single VLM call: ask Gemini which keyboard layout is shown in an image.

How to run:
    # Capture a frame from the camera (CAMERA_INDEX) and ask the model:
    python detect_keyboard_layout.py

    # Skip the camera and use an existing image file:
    python detect_keyboard_layout.py --image outputs/captured_images/opencv__dev_video4.png

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment or .env file.

Returns JSON like:
    {
      "layout": "QWERTY",
      "confidence": 0.95,
      "evidence": "Top row reads Q-W-E-R-T-Y; key to right of L is semicolon."
    }
"""

import argparse
import io
import json
import os
import time

from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

CAMERA_INDEX = 4
MODEL = "models/gemini-3-flash-preview"
FALLBACK_IMAGE = "outputs/captured_images/opencv__dev_video4.png"
TARGET_SIZE = (640, 480)

LAYOUT_PROMPT = """You are looking at a photo of a physical computer keyboard.
Identify which keyboard layout it uses. Choose ONE of:
QWERTY, QWERTZ, AZERTY, DVORAK, COLEMAK, OTHER.

Look at:
  - The top letter row (QWERTY -> Q W E R T Y, QWERTZ -> Q W E R T Z,
    AZERTY -> A Z E R T Y).
  - The position of Y vs Z, and of A vs Q.
  - Any special keys that hint at locale (e.g., Ä Ö Ü -> German QWERTZ,
    Ç -> French AZERTY).

Return strictly this JSON, no extra text:
{
  "layout": "QWERTY" | "QWERTZ" | "AZERTY" | "DVORAK" | "COLEMAK" | "OTHER",
  "confidence": number between 0 and 1,
  "evidence": "short sentence describing what you saw that determined the layout"
}"""


def capture_frame() -> Image.Image:
    try:
        from lerobot.cameras.configs import ColorMode, Cv2Rotation
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        config = OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX,
            fps=30,
            width=TARGET_SIZE[0],
            height=TARGET_SIZE[1],
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
        )
        with OpenCVCamera(config) as camera:
            frame = camera.read()
        return Image.fromarray(frame)
    except Exception as e:
        print(f"Camera unavailable ({e!s}); using fallback image: {FALLBACK_IMAGE}")
        return Image.open(FALLBACK_IMAGE).convert("RGB")


def image_to_part(image: Image.Image) -> types.Part:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None,
                        help="Skip the camera and use this image file instead.")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set the GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable."
        )

    image = Image.open(args.image).convert("RGB") if args.image else capture_frame()
    if image.size != TARGET_SIZE:
        print(f"Resizing image from {image.size} to {TARGET_SIZE}.")
        image = image.resize(TARGET_SIZE, Image.LANCZOS)

    client = genai.Client(api_key=api_key)
    image_part = image_to_part(image)

    start = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL,
        contents=[LAYOUT_PROMPT, image_part],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    elapsed = time.perf_counter() - start
    print(f"[layout] Google API call took {elapsed:.2f}s")

    data = json.loads(response.text)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
