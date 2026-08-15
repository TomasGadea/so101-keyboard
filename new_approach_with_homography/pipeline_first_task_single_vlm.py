"""
First task pipeline (single VLM call variant):

Captures one image, then asks the VLM for SPACE, ENTER, R, L in a single call,
then drives the robot to all four targets consecutively.

Run:
    python new_approach_with_homography/pipeline_first_task_single_vlm.py --config pipeline_config_task1.json
    python new_approach_with_homography/pipeline_first_task_single_vlm.py --config pipeline_config_task1.json --ocr-strong --no-hold
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PROJECT = Path(__file__).resolve().parents[1]
# Insert 3d_coordinates first, then new_approach_with_homography, so this
# directory's modules (e.g. vlm_keyboard_coords.py) take sys.path precedence.
sys.path.insert(0, str(PROJECT / "3d_coordinates"))
sys.path.insert(0, str(PROJECT / "new_approach_with_homography"))

PORT = os.getenv("PORT")
PROJECT_PATH = os.getenv("PROJECT_PATH")
CALIBRATION_DIR = Path(f"{PROJECT_PATH}/new_calibration")
CALIBRATION_ID = "calibration_follower"
URDF = f"{PROJECT_PATH}/new_calibration/so101_new_calib.urdf"
TARGET_FRAME = "gripper_frame_link"

import take_picture
import vlm_keyboard_coords
import ocr_keyboard_coords
import compute_3d_pos
import go2target
import overlay_script

from google.genai import errors as genai_errors
from lerobot.robots.so_follower.so_follower import SO100Follower
from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.model.kinematics import RobotKinematics

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode


KEYS_OUTPUT_PATH = PROJECT / "new_approach_with_homography" / "keys.json"
KEYS_FALLBACK_PATH = KEYS_OUTPUT_PATH
OVERLAY_DIR = PROJECT / "new_approach_with_homography" / "overlays"
OVERLAY_TIME_DIR = PROJECT / "new_approach_with_homography" / "overlay_time"

CAMERA_PATH = int(os.environ["CAMERA_INDEX"])

TASK_KEYS = ["SPACE", "ENTER", "R", "L"]


def build_task_prompt(keys):
    keys_list = ", ".join(f'"{k}"' for k in keys)
    entries = ",\n    ".join(
        f'{{"char": "{k}", "x": number, "y": number}}' for k in keys
    )
    return f"""Analyze the image. Your task is to detect and provide the normalized coordinates for the center of each of the following keys on the keyboard: {keys_list}.
Use a normalized coordinate system where [0, 0] is the top-left and [1000, 1000] is the bottom-right of the image.
Return the data strictly as a JSON object in the following format:

{{
  "letters": [
    {entries}
  ],
  "count": {len(keys)},
  "coordinate_system": "normalized_1000"
}}
Do not provide any conversational text, only the JSON block."""


def run(skip_gemini=False, hold_prompt=True, gemini_model=None,
        use_ocr=False, use_ocr_strong=False, verbose=False, *, config):
    """`config` is the parsed pipeline_config dict, required and passed down
    to compute_3d_pos and go2target."""
    if use_ocr and use_ocr_strong:
        raise ValueError("--ocr and --ocr-strong are mutually exclusive")
    if (use_ocr or use_ocr_strong) and skip_gemini:
        raise ValueError("--ocr/--ocr-strong and --skip-gemini are mutually exclusive")

    keys = [k.upper() for k in TASK_KEYS]

    robot_config = SO100FollowerConfig(
        port=PORT,
        id=CALIBRATION_ID,
        calibration_dir=CALIBRATION_DIR,
    )
    robot = SO100Follower(robot_config)
    robot.connect()

    cam_cfg = OpenCVCameraConfig(
        index_or_path=CAMERA_PATH,
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
        fourcc="MJPG",
        warmup_s=2.0,
    )
    cam = OpenCVCamera(cam_cfg)
    cam.connect()

    bus = FeetechMotorsBus(
        port=PORT,
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }
    )
    bus.connect()

    ik = RobotKinematics(URDF, TARGET_FRAME)

    ocr_engine = None
    if use_ocr or use_ocr_strong:
        print("[pipeline] initializing OCR engine...")
        ocr_init_start = time.perf_counter()
        ocr_engine = ocr_keyboard_coords._get_engine()
        print(f"[pipeline] OCR engine init: {time.perf_counter() - ocr_init_start:.2f}s")

    print("=====STARTING PIPELINE=====")
    pipeline_start = time.perf_counter()

    try:
        # 1. Take picture
        t0 = time.perf_counter()
        image_path = take_picture.run(robot=robot, cam=cam, ik=ik,
                                      verbose=verbose)
        print(f"[pipeline] saved image: {image_path}")
        print(f"[pipeline] capture time: {time.perf_counter() - t0:.2f}s")

        # 2. Locate all task keys
        if use_ocr or use_ocr_strong:
            locate_mode = "ocr-strong" if use_ocr_strong else "ocr"
            print(f"[pipeline] locate mode: {locate_mode}")
            letters_data = ocr_keyboard_coords.run(
                image_path=str(image_path),
                keys=keys,
                keys_json=KEYS_OUTPUT_PATH,
                strong=use_ocr_strong,
                verbose=verbose,
                engine=ocr_engine,
            )
        elif skip_gemini:
            print("[pipeline] locate mode: cached")
            print(f"[pipeline] --skip-gemini: reading cached keys from {KEYS_FALLBACK_PATH}")
            letters_data = json.loads(KEYS_FALLBACK_PATH.read_text())
        else:
            print("[pipeline] locate mode: gemini")
            print(f"[pipeline] Gemini model: {gemini_model or vlm_keyboard_coords.MODEL}")
            prompt = build_task_prompt(keys)
            try:
                letters_data, _ = vlm_keyboard_coords.run(
                    image_path=str(image_path),
                    letters_prompt=prompt,
                    model=gemini_model,
                    verbose=verbose,
                )
            except genai_errors.ServerError as e:
                print(f"[pipeline] Gemini unavailable ({e}); falling back to {KEYS_FALLBACK_PATH}")
                letters_data = json.loads(KEYS_FALLBACK_PATH.read_text())

        pixel_by_char = {entry["char"].upper(): (entry["x"], entry["y"])
                         for entry in letters_data["letters"]}

        KEYS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        KEYS_OUTPUT_PATH.write_text(json.dumps(letters_data, indent=2))

        overlay_path = OVERLAY_DIR / f"{Path(image_path).stem}_overlay.png"
        overlay_script.run(image_path, letters_data, output_path=overlay_path,
                           verbose=verbose)

        OVERLAY_TIME_DIR.mkdir(parents=True, exist_ok=True)
        timestamped_overlay_path = OVERLAY_TIME_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{overlay_path.name}"
        overlay_script.run(image_path, letters_data,
                           output_path=timestamped_overlay_path,
                           verbose=verbose)

        missing = [c for c in keys if c not in pixel_by_char]
        if missing:
            raise ValueError(f"Locator did not return pixels for keys: {missing}")

        targets_uv = [pixel_by_char[c] for c in keys]
        print(f"[pipeline] target pixels (normalized_1000): {dict(zip(keys, targets_uv))}")

        # 3. Pixel -> 3D via homography
        t0 = time.perf_counter()
        target_positions = compute_3d_pos.run(targets_uv, verbose=verbose,
                                              config=config)
        print(f"[pipeline] target 3D positions: "
              f"{dict(zip(keys, [p.tolist() for p in target_positions]))}")
        print(f"[pipeline] 3d computation time: {time.perf_counter() - t0:.2f}s")

        # 4. Drive the robot through all four targets in one go
        typing_start = time.perf_counter()
        go2target.run([list(p) for p in target_positions], verbose=False,
                      robot=robot, ik=ik, bus=bus, config=config)
        typing_elapsed = time.perf_counter() - typing_start

        pipeline_elapsed = time.perf_counter() - pipeline_start
        print(f"[pipeline] typing time: {typing_elapsed:.2f}s")
        print(f"[pipeline] total time:  {pipeline_elapsed:.2f}s")

        if hold_prompt:
            while True:
                if input("Press 'x' + Enter to release the robot: ").strip().lower() == "x":
                    break
    finally:
        robot.disconnect()
        cam.disconnect()
        try:
            bus.disconnect()
        except Exception:
            pass


def pop_value_option(argv, option):
    """Remove an option that accepts either '--name value' or '--name=value'."""
    out = []
    value = None
    i = 0
    prefix = f"{option}="
    while i < len(argv):
        arg = argv[i]
        if arg == option:
            if i + 1 >= len(argv):
                sys.exit(f"{option} requires a value")
            value = argv[i + 1]
            i += 2
            continue
        if arg.startswith(prefix):
            value = arg[len(prefix):]
            i += 1
            continue
        out.append(arg)
        i += 1
    return value, out


if __name__ == "__main__":
    argv = sys.argv[1:]
    gemini_model, argv = pop_value_option(argv, "--gemini-model")
    config_path, argv = pop_value_option(argv, "--config")
    if not config_path:
        sys.exit("--config <pipeline_config.json> is required")
    config = compute_3d_pos.load_config(config_path)

    use_ocr = False
    if "--ocr" in argv:
        use_ocr = True
        argv = [a for a in argv if a != "--ocr"]
    use_ocr_strong = False
    if "--ocr-strong" in argv:
        use_ocr_strong = True
        argv = [a for a in argv if a != "--ocr-strong"]
    skip_gemini = False
    if "--skip-gemini" in argv:
        skip_gemini = True
        argv = [a for a in argv if a != "--skip-gemini"]
    hold_prompt = True
    if "--no-hold" in argv:
        hold_prompt = False
        argv = [a for a in argv if a != "--no-hold"]
    if "--hold" in argv:
        hold_prompt = True
        argv = [a for a in argv if a != "--hold"]
    verbose = False
    if "--verbose" in argv:
        verbose = True
        argv = [a for a in argv if a != "--verbose"]

    if use_ocr and use_ocr_strong:
        sys.exit("--ocr and --ocr-strong are mutually exclusive")
    if (use_ocr or use_ocr_strong) and skip_gemini:
        sys.exit("--ocr/--ocr-strong and --skip-gemini are mutually exclusive")
    if argv:
        sys.exit(f"Unexpected arguments for task 1 pipeline: {' '.join(argv)}")

    run(
        skip_gemini=skip_gemini,
        hold_prompt=hold_prompt,
        gemini_model=gemini_model,
        use_ocr=use_ocr,
        use_ocr_strong=use_ocr_strong,
        verbose=verbose,
        config=config,
    )
