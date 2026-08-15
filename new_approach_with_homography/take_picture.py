"""use /home/till/Documents/robot_learning/robot_learning_project/new_calibration/calibration_follower.json calibration"""

from pathlib import Path
import re
import numpy as np
import time
import threading
import cv2
import io
import os
import json

from PIL import Image
from dotenv import load_dotenv


from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

from lerobot.robots.so_follower.so_follower import SO100Follower
from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig
from lerobot.model.kinematics import RobotKinematics

from dotenv import load_dotenv

load_dotenv()

# Paths for sfm
PROJECT_PATH = os.getenv("PROJECT_PATH")
project_path = Path(f"{PROJECT_PATH}/new_approach_with_homography")
image_path = project_path / "images"

PORT = os.getenv("PORT")
CAMERA_PATH = int(os.environ["CAMERA_INDEX"])
ROBOT_ID = "keyboard_follower"
URDF = f"{PROJECT_PATH}/new_calibration/so101_new_calib.urdf"
CALIBRATION_DIR = Path(f"{PROJECT_PATH}/new_calibration")
CALIBRATION_ID = "calibration_follower"  # loads CALIBRATION_DIR/calibration_follower.json
TARGET_FRAME = "gripper_frame_link"

VERBOSE = False


def vprint(*args, **kwargs):
    """print() that only emits when this module's VERBOSE flag is set."""
    if VERBOSE:
        print(*args, **kwargs)


# this is pre-defined nothing to change here :)
JOINT_CALIBRATION = [
    ["shoulder_pan", 6.0, 1.0],  # Joint 1: zero position offset, scale factor
    ["shoulder_lift", 2.0, 0.97],  # Joint 2: zero position offset, scale factor
    ["elbow_flex", 0.0, 1.05],  # Joint 3: zero position offset, scale factor
    ["wrist_flex", 0.0, 0.94],  # Joint 4: zero position offset, scale factor
    ["wrist_roll", 0.0, 0.5],  # Joint 5: zero position offset, scale factor
    ["gripper", 0.0, 1.0],  # Joint 6: zero position offset, scale factor
]


def apply_joint_calibration(joint_name, raw_position):
    """
    Apply joint calibration coefficients

    Args:
        joint_name: joint name
        raw_position: raw position value

    Returns:
        calibrated_position: calibrated position value
    """
    for joint_cal in JOINT_CALIBRATION:
        if joint_cal[0] == joint_name:
            offset = joint_cal[1]  # zero position offset
            scale = joint_cal[2]  # scale factor
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position  # if no calibration coefficient found, return original value


def take_picture(cam, ik, path_images, best=False):
    picture_start = time.perf_counter()

    read_start = time.perf_counter()
    frame_rgb = cam.async_read(timeout_ms=2000)
    vprint(f"[take_picture] camera async_read: {time.perf_counter() - read_start:.3f}s")

    # Convert to BGR for OpenCV display and saving
    convert_start = time.perf_counter()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    vprint(f"[take_picture] color convert: {time.perf_counter() - convert_start:.3f}s")

    # Read current joint angles
    obs_start = time.perf_counter()
    q_now = np.array([robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors])
    vprint(f"[take_picture] robot observation for filename: {time.perf_counter() - obs_start:.3f}s")

    # Calculate Forward Kinematics (TODO: get the actual position not just the result of FK)
    fk_start = time.perf_counter()
    pose = ik.forward_kinematics(q_now)
    vprint(f"[take_picture] forward kinematics: {time.perf_counter() - fk_start:.3f}s")
    current_x, current_y, current_z = pose[:3, 3]

    # Format filename: x_y_z_timestamp.jpg
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if not best:
        filename = (
            f"coord_x{current_x:.4f}_y{current_y:.4f}_z{current_z:.4f}_{timestamp}.jpg"
        )
    else:
        filename = f"best_view_coord_x{current_x:.4f}_y{current_y:.4f}_z{current_z:.4f}_{timestamp}.jpg"

    # Save the image
    saved_path = path_images / filename
    write_start = time.perf_counter()
    cv2.imwrite(saved_path, frame_bgr)
    vprint(f"[take_picture] image write: {time.perf_counter() - write_start:.3f}s")
    vprint(f"[take_picture] capture helper total: {time.perf_counter() - picture_start:.3f}s")
    return saved_path


def move_to_joint_position(robot, joint_positions, duration=3.0, kp=0.5):
    """
    Use P control to slowly move robot to specified joint positions

    Args:
        robot: robot instance
        joint_positions: dictionary of joint names and target positions
        duration: time to move to target positions (seconds)
        kp: proportional gain
        duration: time to move to zero position (seconds)
        kp: proportional gain
    """
    move_start = time.perf_counter()
    vprint("Using P control to slowly move robot to zero position...")

    # Get current robot state
    current_obs = robot.get_observation()

    # Extract current joint positions
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith(".pos"):
            motor_name = key.removesuffix(".pos")
            current_positions[motor_name] = value

    # Calculate control steps
    control_freq = 150  # 100Hz control frequency
    total_steps = int(duration * 50)
    step_time = 1.0 / control_freq

    vprint(
        f"Will use P control to move to specified positions in {duration} seconds, control frequency: {control_freq}Hz, proportional gain: {kp}"
    )

    for step in range(total_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                # Apply calibration coefficients
                calibrated_value = apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value

        # P control calculation
        robot_action = {}
        for joint_name, target_pos in joint_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos

                # P control: output = Kp * error
                control_output = kp * error

                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position

        # Send action to robot
        if robot_action:
            robot.send_action(robot_action)

        # Show progress
        if step % (control_freq // 2) == 0:  # Show progress every 0.5 seconds
            progress = (step / total_steps) * 100
            vprint(f"Moving to specified positions progress: {progress:.1f}%")

        time.sleep(step_time)

    vprint(f"Robot has moved to specified positions in {time.perf_counter() - move_start:.3f}s")


DEFAULT_POSITIONS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 0.0,
    "wrist_flex": 100.0,
    "wrist_roll": -135.0,
    "gripper": 80.0,
}


def run(joint_positions=DEFAULT_POSITIONS, images_dir=image_path, ask_recalibrate: bool = False,
        robot=None, cam=None, ik=None, verbose=False):
    """
    Connect to robot + camera, move to the given joint positions, and take a picture.

    If `robot` is provided, it is reused and not disconnected (so torque stays
    engaged for downstream callers). Otherwise a robot is created and
    disconnected at the end.
    """
    global VERBOSE
    VERBOSE = verbose

    run_start = time.perf_counter()
    owns_robot = robot is None
    owns_cam = cam is None

    mkdir_start = time.perf_counter()
    images_dir.mkdir(parents=True, exist_ok=True)
    vprint(f"[take_picture.run] ensure images dir: {time.perf_counter() - mkdir_start:.3f}s")

    if owns_robot:
        robot_connect_start = time.perf_counter()
        robot_config = SO100FollowerConfig(
            port=PORT,
            id=CALIBRATION_ID,
            calibration_dir=CALIBRATION_DIR,
        )
        robot = SO100Follower(robot_config)
        robot.connect()
        vprint(f"[take_picture.run] robot connect: {time.perf_counter() - robot_connect_start:.3f}s")
    else:
        vprint("[take_picture.run] robot connect: reused existing robot")

    # expose to module globals so helpers referencing `robot` still work
    globals()["robot"] = robot

    if ik is None:
        ik_start = time.perf_counter()
        ik = RobotKinematics(URDF, TARGET_FRAME)
        vprint(f"[take_picture.run] kinematics init: {time.perf_counter() - ik_start:.3f}s")
    else:
        vprint("[take_picture.run] kinematics init: reused existing kinematics")

    if owns_cam:
        cam_connect_start = time.perf_counter()
        cam_cfg = OpenCVCameraConfig(
            index_or_path=CAMERA_PATH,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
            fourcc="MJPG",
            warmup_s=2,
        )
        cam = OpenCVCamera(cam_cfg)
        cam.connect()
        vprint(f"[take_picture.run] camera connect + warmup: {time.perf_counter() - cam_connect_start:.3f}s")
    else:
        vprint("[take_picture.run] camera connect + warmup: reused existing camera")

    if ask_recalibrate:
        while True:
            calibrate_choice = (
                input("Do you want to recalibrate the robot? (y/n): ").strip().lower()
            )
            if calibrate_choice in ["y", "yes"]:
                vprint("Starting recalibration...")
                robot.calibrate()
                vprint("Calibration completed!")
                break
            elif calibrate_choice in ["n", "no"]:
                vprint("Using previous calibration file")
                break
            else:
                vprint("Please enter y or n")
    else:
        vprint("Using previous calibration file (ask_recalibrate=False)")

    # Apply a hack to always end up in the position where we want to take the image
    # move to this position first, then move close to the target position, then move back to the target position. This seems to be necessary to end up in the right position for taking the picture, otherwise the robot ends up in a slightly different pose where the camera is not looking at the target object.
    close_by_joint_positions = {
        "shoulder_pan": -10.0,
        "shoulder_lift": -30.0,
        "elbow_flex": -10.0,
        "wrist_flex": 90.0,
        "wrist_roll": -122.0,
        "gripper": 80.0,
    }

    move_start = time.perf_counter()
    move_to_joint_position(
        robot, joint_positions=joint_positions, duration=1.0, kp=0.5
    )
    vprint(f"[take_picture.run] move target pose: {time.perf_counter() - move_start:.3f}s")
    
    move_start = time.perf_counter()
    move_to_joint_position(
        robot, joint_positions=close_by_joint_positions, duration=0.75, kp=0.5
    )
    vprint(f"[take_picture.run] move close-by pose: {time.perf_counter() - move_start:.3f}s")
    
    move_start = time.perf_counter()
    move_to_joint_position(
        robot, joint_positions=joint_positions, duration=0.75, kp=0.5
    )
    vprint(f"[take_picture.run] move final pose: {time.perf_counter() - move_start:.3f}s")

    vprint("Reading initial joint angles...")
    obs_start = time.perf_counter()
    start_obs = robot.get_observation()
    vprint(f"[take_picture.run] read initial joint angles: {time.perf_counter() - obs_start:.3f}s")
    start_positions = {}
    for key, value in start_obs.items():
        if key.endswith(".pos"):
            motor_name = key.removesuffix(".pos")
            start_positions[motor_name] = int(value)
            vprint(f"{motor_name}: {value}")

    stabilize_start = time.perf_counter()
    time.sleep(0.1)  # Wait for robot to stabilize
    vprint(f"[take_picture.run] stabilize sleep: {time.perf_counter() - stabilize_start:.3f}s")

    capture_start = time.perf_counter()
    saved_path = take_picture(cam, ik, images_dir, best=True)
    vprint(f"[take_picture.run] take_picture helper: {time.perf_counter() - capture_start:.3f}s")
    vprint(f"Picture saved to: {saved_path}")

    if owns_cam:
        cam_disconnect_start = time.perf_counter()
        cam.disconnect()
        vprint(f"[take_picture.run] camera disconnect: {time.perf_counter() - cam_disconnect_start:.3f}s")
    if owns_robot:
        # Standalone use: hold pose until user releases, then disconnect.
        vprint("Robot is holding position. Press 'c' + Enter to release and continue.")

        stop_event = threading.Event()

        def _hold():
            while not stop_event.is_set():
                time.sleep(0.05)

        holder = threading.Thread(target=_hold, daemon=True)
        holder.start()
        try:
            while True:
                if input().strip().lower() == "c":
                    break
        finally:
            stop_event.set()
            holder.join(timeout=1.0)
        robot.disconnect()
    # When robot was passed in, leave it connected and powered so the
    # caller can keep using it without the arm dropping.
    vprint(f"[take_picture.run] total: {time.perf_counter() - run_start:.3f}s")
    return saved_path


if __name__ == "__main__":
    run()
