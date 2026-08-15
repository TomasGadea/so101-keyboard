#!/usr/bin/env python3
"""
Legacy/debug script. The runtime typing pipeline now uses:
  runtime/type_text.py

Simplified keyboard control for SO100/SO101 robot
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles

https://github.com/Vector-Wangel/XLeRobot/blob/main/software/examples/1_so100_keyboard_ee_control.py
"""

import time
import logging
import traceback
import math
import argparse
import json

import os
import sys
import time
from glob import glob

import numpy as np
from dotenv import load_dotenv

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

load_dotenv()
def resolve_robot_port() -> str:
    configured_port = os.environ.get("PORT") or os.environ.get("FOLLOWER_PORT")
    if configured_port and os.path.exists(configured_port):
        return configured_port

    candidates = sorted(glob("/dev/tty.usbmodem*") + glob("/dev/tty.usbserial*"))
    if candidates:
        if configured_port:
            print(f"Configured port {configured_port!r} does not exist; using {candidates[0]!r}.")
        return candidates[0]

    return configured_port or "/dev/ttyACM0"


PORT = resolve_robot_port()
ROBOT_ID = os.environ.get("ROBOT_ID", "keyboard_follower")
URDF = os.environ["URDF"]
TARGET_FRAME = "gripper_frame_link"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TARGETS_PATH = os.path.join(ROOT, "runtime", "current_key_targets_3d.json")
DEFAULT_PRESS_CONFIG_PATH = os.path.join(ROOT, "calibration", "press_config.json")
TARGET_POSITION = [0.43454982, 0.02153388, 0.12503076]  # Target (x, y+end_effector_diff, z+end_effector_diff) in meters 0.044
TARGET_POSITION_2 = [0.3002, -0.101, 0.048-0.031]  # Target (x, y+end_effector_diff, z+end_effector_diff) in meters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_calibration_ranges(calibration: dict) -> None:
    too_small = []
    for motor, cal in calibration.items():
        span = int(cal.range_max) - int(cal.range_min)
        min_span = 500 if motor != "gripper" else 200
        if span < min_span:
            too_small.append((motor, span, min_span))

    if too_small:
        details = "\n".join(
            f"  - {motor}: range span {span} ticks, expected at least {min_span}"
            for motor, span, min_span in too_small
        )
        raise RuntimeError(
            "Calibration ranges are too small, so motor commands will barely move.\n"
            f"{details}\n"
            "Replace the cached calibration with calibration/keyboard_follower.json, then restart."
        )


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def char_to_key(ch: str) -> str:
    if ch == " ":
        return "SPACE"
    if ch == "\n":
        return "ENTER"
    if ch.isalpha() and len(ch) == 1:
        return ch.upper()
    name = ch.upper()
    if name in {"SPACE", "ENTER"}:
        return name
    raise ValueError(f"Unsupported key/text character: {ch!r}")

# Joint calibration coefficients - manually edited
# Format: [joint_name, zero_position_offset(degrees), scale_factor]
JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],      # Joint 1: zero position offset, scale factor
    ['shoulder_lift', 2.0, 0.97],     # Joint 2: zero position offset, scale factor
    ['elbow_flex', 0.0, 1.05],        # Joint 3: zero position offset, scale factor
    ['wrist_flex', 0.0, 0.94],        # Joint 4: zero position offset, scale factor
    ['wrist_roll', 0.0, 0.5],        # Joint 5: zero position offset, scale factor
    ['gripper', 0.0, 1.0],           # Joint 6: zero position offset, scale factor
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
            scale = joint_cal[2]   # scale factor
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position  # if no calibration coefficient found, return original value

def get_joint_positions(robot) -> np.ndarray:
    return np.array([robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors], dtype=float)


def get_ee_position(ik, robot) -> np.ndarray:
    return np.asarray(ik.forward_kinematics(get_joint_positions(robot))[:3, 3], dtype=float)


def inverse_kinematics(ik, robot, x: float, y: float, z: float, seed_q: np.ndarray | None = None) -> dict:
    q_now = get_joint_positions(robot)
    q_seed = q_now.copy() if seed_q is None else seed_q.copy()

    pose = ik.forward_kinematics(q_seed)
    pose[:3, 3] = [x, y, z]

    # Placo only performs one solve per call. Repeat with the previous solution
    # as the next seed so the joint target actually converges in Cartesian space.
    for _ in range(12):
        q_seed = ik.inverse_kinematics(
            q_seed,
            pose,
            position_weight=1.0,
            orientation_weight=0.0,
        )

    action = {}
    for i, joint_name in enumerate(ik.joint_names):
        action[f"{joint_name}.pos"] = float(q_seed[i])

    if "gripper" in robot.bus.motors:
        action["gripper.pos"] = float(q_now[list(robot.bus.motors).index("gripper")])

    return action, q_seed

def move_to_zero_position(robot, duration=3.0, kp=0.5):
    """
    Use P control to slowly move robot to zero position
    
    Args:
        robot: robot instance
        duration: time to move to zero position (seconds)
        kp: proportional gain
    """
    print("Using P control to slowly move robot to zero position...")
    
    # Get current robot state
    current_obs = robot.get_observation()
    
    # Extract current joint positions
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            current_positions[motor_name] = value
    
    # Zero position targets
    zero_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 60.0,
        'wrist_roll': -135.0,
        'gripper': 80.0
    }
    
    # Calculate control steps
    control_freq = 50  # 50Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq
    
    print(f"Will use P control to move to zero position in {duration} seconds, control frequency: {control_freq}Hz, proportional gain: {kp}")
    
    for step in range(total_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                current_positions[motor_name] = value
        
        # P control calculation
        robot_action = {}
        for joint_name, target_pos in zero_positions.items():
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
            print(f"Moving to zero position progress: {progress:.1f}%")
        
        time.sleep(step_time)
    
    print("Robot has moved to zero position")


def generate_quintic_spline_waypoints(start, end, num_points):

    """
    Args:
        start (np.ndarray): Starting waypoint.
        end (np.ndarray): Ending waypoint.
        num_points (int): Number of points in the trajectory.
        
    Returns:
        np.ndarray: Generated waypoints.
    """
    def f(s):
        return 10 * s**3 - 15 * s**4 + 6 * s**5
    s = np.linspace(0, 1, num_points)
    f_s = f(s)
    waypoints = start + (end - start) * f_s[:, np.newaxis]
    return waypoints

def move2target_position(robot, ik, current_position, target_position, duration=3.0):
    """
    Move robot end-effector to target position using P control
    
    Args:
        robot: robot instance
        ik: inverse kinematics instance
        current_position: current (x, y, z) position of end-effector
        target_position: target (x, y, z) position of end-effector
        duration: time to move to target position (seconds)
        kp: proportional gain
    """
    print(f"Using P control to move robot end-effector to target position {target_position}...")
    
    # Generate waypoints using quintic spline
    num_points = int(duration * 50)  # 50Hz control frequency
    target_position = np.array(target_position, dtype=float)
    waypoints = generate_quintic_spline_waypoints(np.array(current_position), target_position, num_points)
    q_seed = get_joint_positions(robot)
    
    for i, waypoint in enumerate(waypoints):
        try:
            action, q_seed = inverse_kinematics(ik, robot, *waypoint, seed_q=q_seed)
            robot.send_action(action)
        except Exception as e:
            print(f"Error occurred while sending action: {e}")
        
        # Show progress
        if i % 10 == 0:  # Show progress every 10 steps
            progress = (i / num_points) * 100
            current_xyz = get_ee_position(ik, robot)
            error_mm = np.linalg.norm(target_position - current_xyz) * 1000
            print(f"Moving to target position progress: {progress:.1f}% (remaining {error_mm:.1f} mm)")
        
        time.sleep(0.03)  # Sleep for 20ms between control steps
    
    final_xyz = get_ee_position(ik, robot)
    final_error_mm = np.linalg.norm(target_position - final_xyz) * 1000
    print(
        "Robot move finished: "
        f"x={final_xyz[0]:.4f}, y={final_xyz[1]:.4f}, z={final_xyz[2]:.4f}, "
        f"error={final_error_mm:.1f} mm"
    )


def press_key(robot, ik, key_name: str, key_target: dict, press_cfg: dict, hover_only: bool = False):
    surface_z = float(press_cfg["keyboard_surface_z_m"])
    hover_offset = float(press_cfg.get("hover_offset_m", 0.035))
    press_offset = float(press_cfg.get("press_offset_m", -0.004))
    dwell_s = float(press_cfg.get("dwell_s", 0.12))
    retract_offset = float(press_cfg.get("retract_offset_m", hover_offset))

    xy = np.array([float(key_target["x"]), float(key_target["y"])], dtype=float)
    hover = np.array([xy[0], xy[1], surface_z + hover_offset], dtype=float)
    press = np.array([xy[0], xy[1], surface_z + press_offset], dtype=float)
    retract = np.array([xy[0], xy[1], surface_z + retract_offset], dtype=float)

    current = get_ee_position(ik, robot)
    print(f"{key_name}: hover {hover}, press {press}")
    move2target_position(robot, ik, current, hover, duration=1.0)
    if hover_only:
        return

    current = get_ee_position(ik, robot)
    move2target_position(robot, ik, current, press, duration=0.45)
    time.sleep(dwell_s)
    current = get_ee_position(ik, robot)
    move2target_position(robot, ik, current, retract, duration=0.35)


def planned_keys(text: str) -> list[str]:
    if text.upper() in {"SPACE", "ENTER"}:
        return [text.upper()]
    return [char_to_key(ch) for ch in text]



def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Move SO-101 to calibrated keyboard targets and press keys.")
    parser.add_argument("text", nargs="?", default="Q", help="Text/key to press, e.g. Q, A, SPACE, or a short word.")
    parser.add_argument("--targets", default=DEFAULT_TARGETS_PATH)
    parser.add_argument("--press-config", default=DEFAULT_PRESS_CONFIG_PATH)
    parser.add_argument("--hover-only", action="store_true", help="Move above keys without descending to press.")
    parser.add_argument("--skip-zero", action="store_true", help="Do not first move to the default zero pose.")
    parser.add_argument("--press-offset", type=float, default=None, help="Override press offset in meters, e.g. -0.012.")
    args = parser.parse_args()

    print("LeRobot Simplified Keyboard Control Example (P Control)")
    print("="*50)
    
    try:
        # Configure robot
        print(f"Connecting SO-101 follower on {PORT} (id={ROBOT_ID})")
        robot_config = SOFollowerRobotConfig(port=PORT, id=ROBOT_ID, use_degrees=True)
        robot = SO101Follower(robot_config)
        validate_calibration_ranges(robot.calibration)

        # Connect devices
        robot.connect()
        
        print("Device connection successful!")
        
        # Ask whether to recalibrate
        while True:
            calibrate_choice = input("Do you want to recalibrate the robot? (y/n): ").strip().lower()
            if calibrate_choice in ['y', 'yes']:
                print("Starting recalibration...")
                robot.calibrate()
                print("Calibration completed!")
                break
            elif calibrate_choice in ['n', 'no']:
                print("Using previous calibration file")
                break
            else:
                print("Please enter y or n")
        
        # Read initial joint angles
        print("Reading initial joint angles...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                start_positions[motor_name] = int(value)  # Don't apply calibration coefficients
        
        print("Initial joint angles:")
        for joint_name, position in start_positions.items():
            print(f"  {joint_name}: {position}°")
        
        if not os.path.exists(args.targets):
            raise FileNotFoundError(f"Key targets not found: {args.targets}. Run python runtime/build_key_targets.py first.")
        if not os.path.exists(args.press_config):
            raise FileNotFoundError(f"Press config not found: {args.press_config}. Run calibration press-depth setup first.")

        targets = load_json(args.targets)
        press_cfg = load_json(args.press_config)
        if args.press_offset is not None:
            press_cfg["press_offset_m"] = args.press_offset
        key_sequence = planned_keys(args.text)
        missing = [key for key in key_sequence if key not in targets["keys"]]
        if missing:
            raise KeyError(f"Missing keys in {args.targets}: {sorted(set(missing))}")

        print("Planned key sequence:", " ".join(key_sequence))
        for key in key_sequence:
            target = targets["keys"][key]
            print(f"  {key}: x={target['x']:+.4f}, y={target['y']:+.4f}, surface_z={target['z']:+.4f}, pixel={target.get('pixel')}")

        # Move to zero position
        if not args.skip_zero:
            move_to_zero_position(robot, duration=5.0)
        
        
        # Initialize x,y coordinate control
        ik = RobotKinematics(URDF, TARGET_FRAME, joint_names=ARM_JOINTS)
        current = get_ee_position(ik, robot)
        print(f"Initialize end effector position: x={current[0]:.4f}, y={current[1]:.4f}, z={current[2]:.4f}")

        for key in key_sequence:
            press_key(robot, ik, key, targets["keys"][key], press_cfg, hover_only=args.hover_only)

        
        print("Keyboard control instructions:")
        print("x: exit program")
        # Move to the defined position


        while True:
            time.sleep(0.1)  # Main loop sleep to reduce CPU usage
            input_text = input("Press 'x' to exit the program: ").strip()
            if input_text.lower() == 'x':
                print("Exiting program...")
                break
            else:
                print("Invalid input. Please press 'x' to exit.")
        # Disconnect
        robot.disconnect()
        print("Program ended")
        
    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        print("Please check:")
        print("1. Whether the robot is properly connected")
        print("2. Whether the USB port is correct")
        print("3. Whether you have sufficient permissions to access USB devices")
        print("4. Whether the robot is properly configured")

if __name__ == "__main__":
    main() 
