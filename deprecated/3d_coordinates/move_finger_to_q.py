"""
Test VLM function-calling: ask Gemini for a sequence of 1 cm moves that
bring the robot finger (visible in the image) onto the letter Q.

The model is given six tools:
    move_left(), move_right(), move_up(), move_down(),
    move_forward(), move_backward()
each of which moves the finger by exactly 1 cm in the named direction.
"Up/down" are along the image vertical axis, "left/right" along the
horizontal axis, "forward/backward" toward/away from the keyboard plane.

How to run:
    # Live camera frame:
    python move_finger_to_q.py

    # Use an existing image:
    python move_finger_to_q.py --image outputs/captured_images/opencv__dev_video4.png

Requires GEMINI_API_KEY (or GOOGLE_API_KEY).
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

PROMPT = """You see a top-down/angled view of a keyboard with a robot finger
hovering above it. Your job: produce a sequence of function calls that moves
the finger so it ends up directly above the letter Q on the keyboard.

Each available function moves the finger by exactly 1 cm:
  - move_left / move_right : along the image horizontal axis
  - move_up / move_down    : along the image vertical axis
                             (up = away from you in the image, toward the
                             top edge of the frame)
  - move_forward / move_backward : toward / away from the keyboard plane
                                   (forward lowers it onto the key)

Plan first: estimate the current pixel position of the fingertip and of
the Q key, convert the offset to centimeters using the keyboard size as
a reference (a standard letter key is ~1.5 cm wide), then emit only the
move_* function calls in order. Do NOT press down (no move_forward) until
the finger is horizontally above Q. Issue the calls as actual tool calls,
one per 1 cm step, not as text.
"""


def _make_fn(name: str, description: str) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=name,
        description=description,
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    )


TOOLS = [
    types.Tool(function_declarations=[
        _make_fn("move_left",     "Move the robot finger 1 cm to the left."),
        _make_fn("move_right",    "Move the robot finger 1 cm to the right."),
        _make_fn("move_up",       "Move the robot finger 1 cm up (toward top of image)."),
        _make_fn("move_down",     "Move the robot finger 1 cm down (toward bottom of image)."),
        _make_fn("move_forward",  "Move the robot finger 1 cm forward (down toward keyboard surface)."),
        _make_fn("move_backward", "Move the robot finger 1 cm backward (up away from keyboard surface)."),
    ])
]


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


def extract_calls(response) -> list[dict]:
    calls = []
    for cand in response.candidates or []:
        for part in (cand.content.parts or []):
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                calls.append({
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                })
    return calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None,
                        help="Use this image instead of capturing from the camera.")
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
        contents=[PROMPT, image_part],
        config=types.GenerateContentConfig(
            tools=TOOLS,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                ),
            ),
        ),
    )
    elapsed = time.perf_counter() - start
    print(f"[move_to_q] Google API call took {elapsed:.2f}s")

    calls = extract_calls(response)
    if not calls:
        text = getattr(response, "text", None) or ""
        print("Model returned no function calls. Text response:")
        print(text)
        return

    print(f"\nPlanned sequence ({len(calls)} steps):")
    for i, c in enumerate(calls, 1):
        args_str = ", ".join(f"{k}={v}" for k, v in c["args"].items())
        print(f"  {i:3d}. {c['name']}({args_str})")

    summary = {"steps": calls, "count": len(calls)}
    with open("move_to_q_plan.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote move_to_q_plan.json")


if __name__ == "__main__":
    main()
