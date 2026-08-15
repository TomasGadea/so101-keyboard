"""
Pipeline: run the VLM keyboard-coords detection, then draw the overlay.

How to run:
    # All 26 letters (default prompt) + overlay:
    python pipeline.py

    # Use an existing image instead of the camera:
    python pipeline.py --image outputs/captured_images/opencv__dev_video4.png

    # Ask the VLM only for a single key (e.g. "Q"):
    python pipeline.py --key Q

    # Pass a fully custom prompt to the VLM:
    python pipeline.py --prompt "Return JSON ..."

    # Custom output paths:
    python pipeline.py --keys-json keys.json --overlay-out keyboard_overlay.png
"""

import argparse

import vlm_keyboard_coords as vlm
import overlay_script


def build_prompt(args) -> str | None:
    if args.prompt is not None:
        return args.prompt
    if args.key is not None:
        return vlm.SINGLE_KEY_PROMPT_TEMPLATE.format(key=args.key.upper())
    return None  # use default LETTERS_PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None,
                        help="Use this image file instead of capturing from camera.")
    parser.add_argument("--save", default=None,
                        help="Optional path to save the captured frame as a PNG.")
    parser.add_argument("--prompt", default=None,
                        help="Custom prompt for the VLM. Overrides --key.")
    parser.add_argument("--key", default=None,
                        help="Ask the VLM for only this single key (e.g. Q).")
    parser.add_argument("--keys-json", default=vlm.KEYS_JSON,
                        help="Output path for detected key coordinates JSON.")
    parser.add_argument("--overlay-out", default="keyboard_overlay.png",
                        help="Output path for the overlay image.")
    parser.add_argument("--width", type=int, default=vlm.TARGET_SIZE[0],
                        help="Resize image to this width before sending to the VLM.")
    parser.add_argument("--height", type=int, default=vlm.TARGET_SIZE[1],
                        help="Resize image to this height before sending to the VLM.")
    args = parser.parse_args()

    prompt = build_prompt(args)

    letters_data, _ = vlm.run(
        image_path=args.image,
        save_path=args.save,
        keys_json=args.keys_json,
        letters_prompt=prompt,
        target_size=(args.width, args.height),
    )

    overlay_image = args.image if args.image else overlay_script.DEFAULT_IMAGE_PATH
    overlay_script.run(
        image_path=overlay_image,
        keys_data=letters_data,
        corners_data=None,
        output_path=args.overlay_out,
    )


if __name__ == "__main__":
    main()
