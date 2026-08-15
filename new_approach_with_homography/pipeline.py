"""
End-to-end pipeline:

  1. take_picture.run()                -> capture image from camera
  2. vlm_keyboard_coords.run(image)    -> detect letter pixels via Gemini VLM
  3. compute_3d_pos.run(targets)       -> project pixels to 3D via homography
  4. go2target.run(target_positions)   -> move the robot to those 3D points

Run:
    python new_approach_with_homography/pipeline.py --config pipeline_config_task1.json A B
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


VERBOSE = False


def vprint(*args, **kwargs):
    """print() that only emits when this module's VERBOSE flag is set."""
    if VERBOSE:
        print(*args, **kwargs)


def unique_preserving_order(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


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


def make_robot():
    robot_config = SO100FollowerConfig(
        port=PORT,
        id=CALIBRATION_ID,
        calibration_dir=CALIBRATION_DIR,
    )
    robot = SO100Follower(robot_config)
    robot.connect()
    return robot


def make_camera():
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
    return cam


def make_bus():
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
    return bus


def run(letters, skip_gemini=False, hold_prompt=True, gemini_model=None,
        use_ocr=False, use_ocr_strong=False,
        verbose=False, robot=None, cam=None, bus=None, ik=None,
        ocr_deadline_s=None, *, config):
    """Capture, detect, project, and move the robot to the requested letters.

    If robot/cam/bus/ik are provided they're reused and NOT disconnected here;
    the caller owns their lifecycle. If they're None they're created and
    disconnected at the end of this call.

    `config` is the parsed pipeline_config dict; it is required and passed
    down to compute_3d_pos and go2target.
    """
    if use_ocr and use_ocr_strong:
        raise ValueError("--ocr and --ocr-strong are mutually exclusive")
    if (use_ocr or use_ocr_strong) and skip_gemini:
        raise ValueError("--ocr/--ocr-strong and --skip-gemini are mutually exclusive")

    letters = [c.upper() for c in letters]
    requested_keys = unique_preserving_order(letters)
    print(f"[pipeline] key sequence to press ({len(letters)}): {' '.join(letters)}")

    owns_robot = robot is None
    owns_cam = cam is None
    owns_bus = bus is None

    if robot is None:
        robot = make_robot()
    if cam is None:
        cam = make_camera()
    if bus is None:
        bus = make_bus()

    # Warm up the OCR engine before the timed pipeline so its (potentially
    # slow) model load doesn't count against the run.
    ocr_engine = None
    if use_ocr or use_ocr_strong:
        vprint("[pipeline] initializing OCR engine...")
        ocr_init_start = time.perf_counter()
        ocr_engine = ocr_keyboard_coords._get_engine()
        vprint(f"[pipeline] OCR engine init: {time.perf_counter() - ocr_init_start:.2f}s")

    print("=====STARTING PIPELINE=====")
    pipeline_start = time.perf_counter()
    if ik is None:
        ik_start = time.perf_counter()
        ik = RobotKinematics(URDF, TARGET_FRAME)
        vprint(f"[pipeline] shared kinematics init: {time.perf_counter() - ik_start:.2f}s")

    try:
        # 1. Take picture
        time_before_capture = time.perf_counter()
        image_path = take_picture.run(robot=robot, cam=cam, ik=ik, verbose=verbose)
        vprint(f"[pipeline] saved image: {image_path}")
        time_after_capture = time.perf_counter()
        vprint(f"[pipeline] capture time: {time_after_capture - time_before_capture:.2f}s")

        # 2. Letter detection (Gemini by default, local OCR with --ocr)
        if use_ocr or use_ocr_strong:
            locate_mode = "ocr-strong" if use_ocr_strong else "ocr"
            print(f"[pipeline] locate mode: {locate_mode}")
            letters_data = ocr_keyboard_coords.run(
                image_path=str(image_path),
                keys=requested_keys,
                keys_json=KEYS_OUTPUT_PATH,
                strong=use_ocr_strong,
                verbose=verbose,
                engine=ocr_engine,
                deadline_s=ocr_deadline_s,
            )
        elif skip_gemini:
            vprint("[pipeline] locate mode: cached")
            vprint(f"[pipeline] --skip-gemini: reading cached keys from {KEYS_FALLBACK_PATH}")
            letters_data = json.loads(KEYS_FALLBACK_PATH.read_text())
        else:
            vprint("[pipeline] locate mode: gemini")
            vprint(f"[pipeline] Gemini model: {gemini_model or vlm_keyboard_coords.MODEL}")
            letters_prompt = build_requested_keys_prompt(requested_keys)
            try:
                letters_data, _ = vlm_keyboard_coords.run(
                    image_path=str(image_path),
                    letters_prompt=letters_prompt,
                    model=gemini_model,
                    verbose=verbose,
                )
            except genai_errors.ServerError as e:
                vprint(f"[pipeline] Gemini unavailable ({e}); "
                      f"falling back to cached {KEYS_FALLBACK_PATH}")
                letters_data = json.loads(KEYS_FALLBACK_PATH.read_text())
        pixel_by_char = {entry["char"].upper(): (entry["x"], entry["y"])
                         for entry in letters_data["letters"]}

        KEYS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        KEYS_OUTPUT_PATH.write_text(json.dumps(letters_data, indent=2))

        overlay_path = OVERLAY_DIR / f"{Path(image_path).stem}_overlay.png"
        overlay_script.run(image_path, letters_data, output_path=overlay_path, verbose=verbose)

        OVERLAY_TIME_DIR.mkdir(parents=True, exist_ok=True)
        timestamped_overlay_path = OVERLAY_TIME_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{overlay_path.name}"
        overlay_script.run(image_path, letters_data, output_path=timestamped_overlay_path, verbose=verbose)

        missing = [c for c in letters if c not in pixel_by_char]
        if missing:
            raise ValueError(f"Locator did not return pixels for letters: {missing}")

        targets_uv = [pixel_by_char[c] for c in letters]
        vprint(f"[pipeline] target pixels (normalized_1000): "
              f"{dict(zip(letters, targets_uv))}")

        time_before_3d = time.perf_counter()
        # 3. Pixel -> 3D via homography
        target_positions = compute_3d_pos.run(targets_uv, verbose=verbose, config=config)
        vprint(f"[pipeline] target 3D positions: "
              f"{dict(zip(letters, [p.tolist() for p in target_positions]))}")
        vprint(f"[pipeline] 3d computation time: {time.perf_counter() - time_before_3d:.2f}s")

        # 4. Drive the robot (reuse the same robot so it stays held)
        typing_start = time.perf_counter()
        go2target.run([list(p) for p in target_positions], verbose=verbose, robot=robot, ik=ik, bus=bus, config=config)
        typing_elapsed = time.perf_counter() - typing_start
        
        pipeline_elapsed = time.perf_counter() - pipeline_start
        vprint(f"[pipeline] typing time:   {typing_elapsed:.2f}s")
        print(f"[pipeline] total time:    {pipeline_elapsed:.2f}s")

        # Hold the robot in place until the user releases it.
        if hold_prompt:
            while True:
                if input("Press 'x' + Enter to release the robot: ").strip().lower() == "x":
                    break
    finally:
        if owns_robot:
            robot.disconnect()
        if owns_cam:
            cam.disconnect()
        if owns_bus:
            try:
                bus.disconnect()
            except Exception:
                pass


SPECIAL_KEYS = {"SPACE", "ENTER"}


def expand_args(argv):
    """Expand multi-char args (e.g. "hello world") into per-character keys.

    Single chars and recognized special key names (SPACE, ENTER) are kept as-is.
    Whitespace inside a multi-char arg becomes the SPACE key.
    """
    out = []
    for a in argv:
        if len(a) <= 1 or a.upper() in SPECIAL_KEYS:
            out.append(a)
            continue
        for ch in a:
            if ch.isspace():
                out.append("SPACE")
            else:
                out.append(ch)
    return out


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
    ocr_deadline_s, argv = pop_value_option(argv, "--ocr-deadline-s")
    if ocr_deadline_s is not None:
        try:
            ocr_deadline_s = float(ocr_deadline_s)
        except ValueError:
            sys.exit("--ocr-deadline-s requires a numeric value")
    if not config_path:
        sys.exit("--config <pipeline_config.json> is required")
    config = json.loads(Path(config_path).read_text())
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
    if use_ocr and use_ocr_strong:
        sys.exit("--ocr and --ocr-strong are mutually exclusive")
    if (use_ocr or use_ocr_strong) and skip_gemini:
        sys.exit("--ocr/--ocr-strong and --skip-gemini are mutually exclusive")

    verbose = False
    if "--verbose" in argv:
        verbose = True
        argv = [a for a in argv if a != "--verbose"]
    args = expand_args(argv) if argv else ["Q", "P"]
    run(
        args,
        skip_gemini=skip_gemini,
        use_ocr=use_ocr,
        use_ocr_strong=use_ocr_strong,
        gemini_model=gemini_model,
        hold_prompt=hold_prompt,
        verbose=verbose,
        ocr_deadline_s=ocr_deadline_s,
        config=config)
