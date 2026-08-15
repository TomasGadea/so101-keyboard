#!/usr/bin/env python3
"""
Simplified keyboard control for SO100/SO101 robot
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles

https://github.com/Vector-Wangel/XLeRobot/blob/main/software/examples/1_so100_keyboard_ee_control.py
"""

import json
import time
import logging
import traceback
import math

import sys

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.so_follower import SO100Follower
from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
import os
from pathlib import Path

PORT = os.getenv("PORT")
PROJECT_PATH = os.getenv("PROJECT_PATH")
URDF = f"{PROJECT_PATH}/new_calibration/so101_new_calib.urdf"
CALIBRATION_DIR = Path(f"{PROJECT_PATH}/new_calibration")
CALIBRATION_ID = (
    "calibration_follower"  # loads CALIBRATION_DIR/calibration_follower.json
)
TARGET_FRAME = "gripper_frame_link"

def load_config(path):
    """Load a pipeline config from an explicit path (no default)."""
    with open(path, "r") as f:
        return json.load(f)

# TODO
TARGET_POSITION = [
    0.20663391,
    0.10661111,
    0.025,
]  # Target (x, y+end_effector_diff, z+end_effector_diff) in meters 0.044
TARGET_POSITION_2 = [
    0.22819507,
    -0.05924976,
    0.025,
]  # Target (x, y+end_effector_diff, z+end_effector_diff) in meters
# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


VERBOSE = False


def vprint(*args, **kwargs):
    """print() that only emits when this module's VERBOSE flag is set."""
    if VERBOSE:
        print(*args, **kwargs)


# Joint calibration coefficients - manually edited
# Format: [joint_name, zero_position_offset(degrees), scale_factor]
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


def inverse_kinematics(
    ik, robot, x: float, y: float, z: float, finger_tip_pose, verbose: bool = False
) -> None:
    q_now = np.array([robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors])

    pose = ik.forward_kinematics(q_now)
    if verbose:
        vprint(
            f"Current end effector position: x={pose[0, 3]:.4f}, y={pose[1, 3]:.4f}, z={pose[2, 3]:.4f}"
        )
    # `finger_tip_pose` is a 4x4 homogeneous transform whose 3x3 rotation block
    # makes the finger tip face directly downwards. Inject the target (x, y, z)
    # into the translation column.
    pose = finger_tip_pose.copy()
    pose[:3, 3] = [x, y, z]
    q_target = ik.inverse_kinematics(q_now, pose)

    action = {f"{m}.pos": float(q_target[i]) for i, m in enumerate(robot.bus.motors)}
    action["gripper.pos"] = float(q_now[-1])
    return action


def move_to_zero_position(robot, duration=3.0, kp=0.5, verbose: bool = False):
    """
    Use P control to slowly move robot to zero position

    Args:
        robot: robot instance
        duration: time to move to zero position (seconds)
        kp: proportional gain
    """
    if verbose:
        vprint("Using P control to slowly move robot to zero position...")

    # Get current robot state
    current_obs = robot.get_observation()

    # Extract current joint positions
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith(".pos"):
            motor_name = key.removesuffix(".pos")
            current_positions[motor_name] = value

    # Zero position targets
    zero_positions = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 90.0,
        "wrist_roll": -135.0,
        "gripper": 80.0,
    }

    # Calculate control steps
    control_freq = 50  # 50Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq

    if verbose:
        vprint(
            f"Will use P control to move to zero position in {duration} seconds, control frequency: {control_freq}Hz, proportional gain: {kp}"
        )

    pvre_joints_pos = None

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

        if pvre_joints_pos is None:
            pvre_joints_pos = {
                joint: current_positions[joint] for joint in zero_positions.keys()
            }
        else:
            # Check difference from previous step to determine if we are close to the target
            close_enough = True
            for joint_name in zero_positions.keys():
                if joint_name in current_positions and joint_name in pvre_joints_pos:
                    pos_diff = abs(
                        current_positions[joint_name] - pvre_joints_pos[joint_name]
                    )
                    if (
                        pos_diff > 1.0
                    ):  # If any joint is still far from target, continue control
                        close_enough = False
                        break

            if close_enough:
                break

            pvre_joints_pos = {
                joint: current_positions[joint] for joint in zero_positions.keys()
            }

        # Show progress
        if (
            verbose and step % (control_freq // 2) == 0
        ):  # Show progress every 0.5 seconds
            progress = (step / total_steps) * 100
            vprint(f"Moving to zero position progress: {progress:.1f}%")

        time.sleep(step_time)

    if verbose:
        vprint("Robot has moved to zero position")


def generate_linear_waypoints(start, end, num_points):
    """
    Args:
        start (np.ndarray): Starting waypoint.
        end (np.ndarray): Ending waypoint.
        num_points (int): Number of points in the trajectory.

    Returns:
        np.ndarray: Generated waypoints.
    """
    s = np.linspace(0, 1, num_points)
    waypoints = start + (end - start) * s[:, np.newaxis]
    return waypoints


def move2target_position(
    robot,
    ik,
    current_position,
    target_position,
    finger_tip_pose,
    num_keypoints=250,
    verbose: bool = False,
    target=False,
    bus=None,
    auto_stop=None,
):
    """
    Move robot end-effector to target position using P control

    Args:
        robot: robot instance
        ik: inverse kinematics instance
        current_position: current (x, y, z) position of end-effector
        target_position: target (x, y, z) position of end-effector
        finger_tip_pose: 4x4 transform forcing the finger tip to face downwards
        num_keypoints: number of keypoints for the trajectory
        auto_stop: pipeline_config "auto_stop" dict (early-termination settings)
        kp: proportional gain
    """
    if auto_stop is None:
        auto_stop = {}
    if verbose:
        vprint(
            f"Using P control to move robot end-effector to target position {target_position}..."
        )

    # Generate waypoints using linear interpolation
    num_points = num_keypoints
    waypoints = generate_linear_waypoints(
        np.array(current_position), np.array(target_position), num_points
    )

    if target:
        counter = 0
        prev_dist_error = float(0)
        acc_load = []
        z_history = []
        x_history = []
        y_history = []
        key_pressed = False
    for i, waypoint in enumerate(waypoints):
        try:
            action = inverse_kinematics(
                ik, robot, *waypoint, finger_tip_pose, verbose=verbose
            )
            robot.send_action(action)

            if target and auto_stop.get("enabled", True):
                
                # Stop early if the z coordinate stops changing, which indicates
                # the finger tip has touched the keyboard.
                obs = robot.get_observation()
                q_now = np.array([obs[f"{m}.pos"] for m in robot.bus.motors])
                pose = ik.forward_kinematics(q_now)
                target_orientation = finger_tip_pose[:3, :3]
                current_orientation = pose[:3, :3]
                orientation_diff = np.linalg.norm(target_orientation - current_orientation)
                # vprint(f"Orientation difference from facing down: {orientation_diff:.4f}")
                step_stop = int(num_points * auto_stop["start_fraction"])
                if orientation_diff < auto_stop["pose_orientation_treshold"] and i >= step_stop:
                    vprint("Orientation distance very low, likely indicates contact with keyboard, stopping movement")
                    key_pressed = True
                    break
                # ee_z = pose[2, 3]
                # z_history.append(ee_z)

                # ee_x = pose[0, 3]
                # ee_y = pose[1, 3]
                # x_history.append(ee_x)
                # y_history.append(ee_y)


                # diff = AUTO_STOP["z_diff_threshold"]
                # vprint(f"Diff to previous z: {abs(z_history[-1] - z_history[-2]) if len(z_history) > 1 else 'N/A'} (threshold: {diff})")
                # vprint(f"Current z: {ee_z:.4f}, target z: {target_position[2]:.4f}")
                # vprint(f"Current x: {ee_x:.4f}, target x: {target_position[0]:.4f}")
                # vprint(f"Current y: {ee_y:.4f}, target y: {target_position[1]:.4f}")
                # n_stalled = AUTO_STOP["consecutive_stalled_steps"]
                # start_idx = int(num_points * AUTO_STOP["start_fraction"])

                # if i >= start_idx and len(z_history) >= n_stalled + 1 +10000000:
                #     dzs = [
                #         abs(z_history[-k] - z_history[-k - 1])
                #         for k in range(1, n_stalled + 1)
                #     ]
                #     if all(d < diff for d in dzs):
                #         vprint(
                #             f"Z position stalled (Δ<{diff} for {n_stalled} steps with Δs={[f'{d:.4f}' for d in dzs]}) at z={ee_z:.4f}m, stopping movement"
                #         )
                #         break

                # # Get observation
                # obs = robot.get_observation()
                # q_now = np.array([obs[f"{m}.pos"] for m in robot.bus.motors])
                # pose = ik.forward_kinematics(q_now)
                # ee_position = pose[:3, 3]

                # vprint(f"Current end effector position: x={ee_position[0]:.4f}, y={ee_position[1]:.4f}, z={ee_position[2]:.4f}")

                # # Position error in z axis
                # dist_error = np.linalg.norm(ee_position - target_position)
                # if dist_error > prev_dist_error:
                #     counter += 1
                #     vprint(f"Distance error increased to {dist_error:.4f}m, counter: {counter}")
                # else:
                #     counter = 0
                #     vprint(f"Distance error decreased to {dist_error:.4f}m, counter reset to 0")
                # prev_dist_error = dist_error

                # # If for five consecutive steps the z error is increasing, we consider that we have reached the target position (It means that we are pressing down the keyboard)
                # if counter >= 2:
                #     vprint(f"Target position reached with distance error {dist_error:.4f}m, stopping movement")
                #     break

                # motor_names = list(bus.motors.keys())

                # total_load = 0
                # for name in motor_names:
                #     # Llegim els registres motor per motor
                #     current = bus.read("Present_Current", name)
                #     load = bus.read("Present_Load", name)

                #     total_load += abs(load)

                # prev_acc_load = sum(acc_load) / len(acc_load) if acc_load else 0
                # avg_load = total_load / len(motor_names)
                # acc_load += [avg_load]

                # curr_acc_load = sum(acc_load[-3:]) / 3 if len(acc_load) >= 5 else 0
                # vprint(f"Average motor load: {avg_load:.2f}, Current accumulated average load: {curr_acc_load:.2f}, Previous accumulated average load: {prev_acc_load:.2f}")
                # # if i > int(num_points * 0.5) and np.any(np.array(acc_load[-3:]) > prev_acc_load*1.05):
                # #     break

                # if counter >= 3:
                #     vprint(f"High load detected for 3 consecutive steps, assuming target position reached, stopping movement")
                #     break

        except Exception as e:
            vprint(f"Error occurred while sending action: {e}")

        # Show progress
        if verbose and i % 10 == 0:  # Show progress every 10 steps
            progress = (i / num_points) * 100
            vprint(f"Moving to target position progress: {progress:.1f}%")

        # time.sleep(0.0015)  # Sleep for 20ms between control steps
    else:
        if target:
            vprint(
                f"Did not break early. Last 10 z values: {[f'{z:.4f}' for z in z_history[-10:]]}"
            )

    if verbose:
        vprint("Robot has moved to target position")
    
    if target and auto_stop.get("enabled", True):
        return key_pressed
    else:
        return True


def run(
    target_positions,
    verbose: bool = False,
    ask_recalibrate: bool = False,
    robot=None,
    ik=None,
    bus=None,
    *,
    config,
):
    """Run the pick-and-place routine for a list of target positions.

    `config` is the parsed pipeline_config dict; it is required and must be
    passed explicitly by the caller. Hover offsets, trajectory resolution,
    auto-stop settings and the finger-tip pose are all read from it.
    """
    global VERBOSE
    VERBOSE = verbose

    hover = config["hover"]
    num_keypoints_cfg = config["num_keypoints"]
    auto_stop = config["auto_stop"]
    finger_tip_pose = np.array(config["finger_tip_pose_facing_down"], dtype=float)

    if verbose:
        vprint("LeRobot Simplified Keyboard Control Example (P Control)")
        vprint("=" * 50)

    try:
        # Get port
        # port = input("Please enter the USB port for SO100 robot (e.g., /dev/ttyACM0): ").strip()

        # # If directly press Enter, use default port
        # if not port:
        #     port = "/dev/ttyACM0"
        #     vprint(f"Using default port: {port}")
        # else:
        #     vprint(f"Connecting to port: {port}")

        owns_robot = robot is None
        if owns_robot:
            robot_config = SO100FollowerConfig(
                port=PORT,
                id=CALIBRATION_ID,
                calibration_dir=CALIBRATION_DIR,
            )
            robot = SO100Follower(robot_config)
            robot.connect()

        # Configure keyboard
        keyboard_config = KeyboardTeleopConfig()
        keyboard = KeyboardTeleop(keyboard_config)
        keyboard.connect()

        if verbose:
            vprint("Device connection successful!")

        # Ask whether to recalibrate (only when explicitly requested)
        if ask_recalibrate:
            while True:
                calibrate_choice = (
                    input("Do you want to recalibrate the robot? (y/n): ")
                    .strip()
                    .lower()
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
            if verbose:
                vprint("Using previous calibration file (ask_recalibrate=False)")

        # Read initial joint angles
        if verbose:
            vprint("Reading initial joint angles...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                start_positions[motor_name] = int(
                    value
                )  # Don't apply calibration coefficients

        if verbose:
            vprint("Initial joint angles:")
            for joint_name, position in start_positions.items():
                vprint(f"  {joint_name}: {position}°")

        # # Move to zero position
        # move_to_zero_position(robot, duration=1.0, verbose=verbose)

        # Initialize x,y coordinate control (TODO)
        if ik is None:
            ik_start = time.perf_counter()
            ik = RobotKinematics(URDF, TARGET_FRAME)
            vprint(
                f"[go2target.run] kinematics init: {time.perf_counter() - ik_start:.3f}s"
            )
        else:
            vprint("[go2target.run] kinematics init: reused existing kinematics")
        q_now = np.array(
            [robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors]
        )

        pose = ik.forward_kinematics(q_now)
        current_x, current_y, current_z = pose[:3, 3]
        # vprint(f"Initial end effector position: x={current_x:.4f}, y={current_y:.4f}, z={current_z:.4f}")
        # exit(0)

        if verbose:
            vprint(
                f"Initialize end effector position: x={current_x:.4f}, y={current_y:.4f}, z={current_z:.4f}"
            )

        # hover offsets from the pipeline config passed in by the caller
        pre = hover["pre_press"]
        post = hover["post_press"]

        for idx, target_position in enumerate(target_positions):
            pre_hover = [
                target_position[0] + pre["x_offset"],
                target_position[1] * pre["y_mult"],
                target_position[2] + pre["z_offset"],
            ]
            post_hover = [
                target_position[0] + post["x_offset"],
                target_position[1] * post["y_mult"],
                target_position[2] + post["z_offset"],
            ]

            q_now = np.array(
                [robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors]
            )
            pose = ik.forward_kinematics(q_now)
            current_x, current_y, current_z = pose[:3, 3]

            # Distance from current position to target position
            dist_to_target = np.linalg.norm(
                np.array([current_x, current_y, current_z]) - np.array(pre_hover)
            )
            vprint(f"[target {idx}] Distance to target: {dist_to_target:.4f}")

            num_keypoints = num_keypoints_cfg

            if verbose:
                vprint(
                    f"[target {idx}] Moving above target from: x={current_x:.4f}, y={current_y:.4f}, z={current_z:.4f}"
                )
            move2target_position(
                robot,
                ik,
                (current_x, current_y, current_z),
                pre_hover,
                finger_tip_pose,
                num_keypoints=max(75, int(num_keypoints * (dist_to_target / 0.10))),
                verbose=verbose,
            )

            q_now = np.array(
                [robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors]
            )
            pose = ik.forward_kinematics(q_now)
            current_x, current_y, current_z = pose[:3, 3]
            if verbose:
                vprint(
                    f"[target {idx}] Descending to target from: x={current_x:.4f}, y={current_y:.4f}, z={current_z:.4f}"
                )

            key_pressed = move2target_position(
                robot,
                ik,
                (current_x, current_y, current_z),
                target_position,
                finger_tip_pose,
                num_keypoints=int(num_keypoints * 1.2),
                verbose=verbose,
                target=True,
                bus=bus,
                auto_stop=auto_stop,
            )
            if not key_pressed:
                vprint(f"[target {idx}] Warning: did not detect key press based on orientation, may not have successfully pressed the key")
                vprint(f"[target {idx}] Attempting to press the key again with a direct move to target position")
                q_now = np.array(
                    [robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors]
                )
                pose = ik.forward_kinematics(q_now)
                current_x, current_y, current_z = pose[:3, 3]
                move2target_position(
                    robot,
                    ik,
                    (current_x, current_y, current_z),
                    (target_position[0]-0.01, target_position[1], config["plane_z_2"]),
                    finger_tip_pose,
                    num_keypoints=int(num_keypoints * 1),
                    verbose=verbose,
                    target=True,
                    bus=bus,
                    auto_stop=auto_stop,
                )
            time.sleep(0.1)

            # Skip the post-press retreat on the final key so the
            # end-effector stays resting on it.
            if idx < len(target_positions) - 1:
                q_now = np.array(
                    [robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors]
                )
                pose = ik.forward_kinematics(q_now)
                current_x, current_y, current_z = pose[:3, 3]
                if verbose:
                    vprint(
                        f"[target {idx}] Retreating above target from: x={current_x:.4f}, y={current_y:.4f}, z={current_z:.4f}"
                    )
                move2target_position(
                    robot,
                    ik,
                    (current_x, current_y, current_z),
                    post_hover,
                    finger_tip_pose,
                    num_keypoints=num_keypoints,
                    verbose=verbose,
                )

        if verbose:
            vprint("Keyboard control instructions:")
            vprint("x: exit program")
        # Move to the defined position

        if owns_robot:
            while True:
                time.sleep(0.1)  # Main loop sleep to reduce CPU usage
                input_text = input("Press 'x' to exit the program: ").strip()
                if input_text.lower() == "x":
                    vprint("Exiting program...")
                    break
                else:
                    vprint("Invalid input. Please press 'x' to exit.")
            robot.disconnect()
        if verbose:
            vprint("Program ended")

    except Exception as e:
        vprint(f"Program execution failed: {e}")
        traceback.print_exc()
        vprint("Please check:")
        vprint("1. Whether the robot is properly connected")
        vprint("2. Whether the USB port is correct")
        vprint("3. Whether you have sufficient permissions to access USB devices")
        vprint("4. Whether the robot is properly configured")


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <pipeline_config.json>")
    run([TARGET_POSITION, TARGET_POSITION_2], config=load_config(sys.argv[1]))


if __name__ == "__main__":
    main()
