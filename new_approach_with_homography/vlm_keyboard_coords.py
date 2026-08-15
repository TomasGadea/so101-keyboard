"""
Run two VLM calls on a keyboard image:

  1. Detect all 26 letter keys -> writes keys.json.
  2. Detect the 4 keyboard corners -> writes corners.json.

Both calls return coordinates in the "normalized_1000" system.

How to run:
    # Capture a frame from the camera (CAMERA_INDEX) and run both VLM calls:
    python vlm_keyboard_coords.py

    # Skip the camera and use an existing image file:
    python vlm_keyboard_coords.py --image outputs/captured_images/opencv__dev_video4.png

    # Save the captured/used frame as a PNG:
    python vlm_keyboard_coords.py --save frame.png

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment or .env file.
Set GEMINI_MODEL to override the default model.
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

CAMERA_INDEX = os.environ.get("CAMERA_INDEX")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
FALLBACK_IMAGE = "outputs/captured_images/opencv__dev_video4.png"
KEYS_JSON = "keys.json"
CORNERS_JSON = "corners.json"
TARGET_SIZE = (640, 480)

VERBOSE = False


def vprint(*args, **kwargs):
    """print() that only emits when this module's VERBOSE flag is set."""
    if VERBOSE:
        print(*args, **kwargs)

LETTERS_PROMPT = """Analyze the image opencv__dev_video4.png. Your task is to detect and provide normalized coordinates for every individual letter key on the keyboard (A through Z).
For each letter, identify the center point of the key. Use a normalized coordinate system where $[0, 0]$ is the top-left and $[1000, 1000]$ is the bottom-right of the image.
Return the data strictly as a JSON object where each key is the letter and the value is an object containing the coordinates. Follow this format:
JSON

{
  "letters": [
    {"char": "Q", "x": number, "y": number},
    {"char": "W", "x": number, "y": number},
    ...
  ],
  "count": 26,
  "coordinate_system": "normalized_1000"
}
Ensure all 26 letters are included. Do not provide any conversational text, only the JSON block."""

LETTERS_AND_SPACE_PROMPT = """Analyze the image opencv__dev_video4.png. Your task is to detect and provide normalized coordinates for every individual letter key on the keyboard (A through Z) AND the SPACE bar.
For each key, identify the center point of the key. Use a normalized coordinate system where $[0, 0]$ is the top-left and $[1000, 1000]$ is the bottom-right of the image.
Return the data strictly as a JSON object where each entry contains the character and its coordinates. Use "SPACE" as the char value for the space bar. Follow this format:
JSON

{
  "letters": [
    {"char": "Q", "x": number, "y": number},
    {"char": "W", "x": number, "y": number},
    ...
    {"char": "SPACE", "x": number, "y": number}
  ],
  "count": 27,
  "coordinate_system": "normalized_1000"
}
Ensure all 26 letters and the SPACE bar are included. Do not provide any conversational text, only the JSON block."""

SINGLE_KEY_PROMPT_TEMPLATE = """Analyze the image. Your task is to detect and provide the normalized coordinates for the center of the single letter key "{key}" on the keyboard.
Use a normalized coordinate system where [0, 0] is the top-left and [1000, 1000] is the bottom-right of the image.
Return the data strictly as a JSON object in the following format:

{{
  "letters": [
    {{"char": "{key}", "x": number, "y": number}}
  ],
  "count": 1,
  "coordinate_system": "normalized_1000"
}}
Do not provide any conversational text, only the JSON block."""

CORNERS_PROMPT = """Analyze the image opencv__dev_video4.png. Your task is to identify the four outer corners of the silver/white wireless keyboard.
Provide the coordinates using a normalized coordinate system where $[0, 0]$ represents the top-left corner and $[1000, 1000]$ represents the bottom-right corner of the image.
Return the data strictly in the following JSON format:

{
  "object": "keyboard",
  "coordinates": {
    "top_left": {"x": number, "y": number},
    "top_right": {"x": number, "y": number},
    "bottom_right": {"x": number, "y": number},
    "bottom_left": {"x": number, "y": number}
  },
  "coordinate_system": "normalized_1000"
}
Do not include any conversational text or explanations, only the JSON block."""

def capture_frame() -> Image.Image:
    try:
        from lerobot.cameras.configs import ColorMode, Cv2Rotation
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        config = OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
        )
        with OpenCVCamera(config) as camera:
            frame = camera.read()
        return Image.fromarray(frame)
    except Exception as e:
        vprint(f"Camera unavailable ({e!s}); using fallback image: {FALLBACK_IMAGE}")
        return Image.open(FALLBACK_IMAGE).convert("RGB")


def image_to_part(image: Image.Image) -> types.Part:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


def call_vlm(client: genai.Client, prompt: str, image_part: types.Part,
             label: str = "vlm", model: str = MODEL) -> dict:
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=[prompt, image_part],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    elapsed = time.perf_counter() - start
    vprint(f"[{label}] {model} took {elapsed:.2f}s")
    return json.loads(response.text)


def run(image_path=None, save_path=None, keys_json=KEYS_JSON, corners_json=CORNERS_JSON,
        letters_prompt=None, target_size=TARGET_SIZE, model=None, verbose=False):
    """
    Run both VLM calls on a keyboard image.

    Args:
        image_path: optional path to an existing image. If None, captures from camera.
        save_path: optional path to save the captured/used frame as a PNG.
        keys_json: output path for the detected letters JSON.
        corners_json: output path for the detected corners JSON.

    Returns:
        (letters_data, corners_data) tuple of dicts parsed from the VLM responses.
    """
    global VERBOSE
    VERBOSE = verbose

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set the GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable."
        )

    image = Image.open(image_path).convert("RGB") if image_path else capture_frame()
    if image.size != tuple(target_size):
        vprint(f"Resizing image from {image.size} to {tuple(target_size)}.")
        image = image.resize(tuple(target_size), Image.LANCZOS)
    if save_path:
        image.save(save_path)

    client = genai.Client(api_key=api_key)
    image_part = image_to_part(image)

    use_model = model or MODEL
    prompt = letters_prompt if letters_prompt is not None else LETTERS_AND_SPACE_PROMPT
    letters_data = call_vlm(client, prompt, image_part, label="letters", model=use_model)
    with open(keys_json, "w") as f:
        json.dump(letters_data, f, indent=2)
    vprint(f"Wrote {len(letters_data.get('letters', []))} letters to {keys_json}")

    # corners_data = call_vlm(client, CORNERS_PROMPT, image_part, label="corners")
    # with open(corners_json, "w") as f:
    #     json.dump(corners_data, f, indent=2)
    # vprint(f"Wrote {len(corners_data.get('coordinates', {}))} corners to {corners_json}")
    corners_data = None

    return letters_data, corners_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None,
                        help="Skip the camera and use this image file instead.")
    parser.add_argument("--save", default=None,
                        help="Optional path to save the captured frame as a PNG.")
    parser.add_argument("--model", default=None,
                        help="Gemini model to use. Overrides GEMINI_MODEL.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress and timing information.")
    args = parser.parse_args()

    run(image_path=args.image, save_path=args.save, model=args.model,
        verbose=args.verbose)


if __name__ == "__main__":
    main()
