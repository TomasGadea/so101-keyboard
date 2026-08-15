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

import os
import sys
import time

import numpy as np
from dotenv import load_dotenv

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

load_dotenv()
PORT = os.environ.get("PORT") or os.environ.get("FOLLOWER_PORT") or "/dev/ttyACM0"
ROBOT_ID = os.environ.get("ROBOT_ID", "keyboard_follower")
URDF = os.environ["URDF"]
TARGET_FRAME = "gripper_frame_link"
TARGET_POSITION = [0.43454982, 0.02153388, 0.12503076]  # Target (x, y+end_effector_diff, z+end_effector_diff) in meters 0.044
TARGET_POSITION_2 = [0.3002, -0.101, 0.048-0.031]  # Target (x, y+end_effector_diff, z+end_effector_diff) in meters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def inverse_kinematics(ik, robot, x: float, y: float, z: float) -> None:
    q_now = np.array([robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors])

    pose = ik.forward_kinematics(q_now)
    pose[:3, 3] = [x, y, z]
    q_target = ik.inverse_kinematics(q_now, pose)

    action = {f"{m}.pos": float(q_target[i]) for i, m in enumerate(robot.bus.motors)}
    action["gripper.pos"] = float(q_now[-1])
    return action

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
    waypoints = generate_quintic_spline_waypoints(np.array(current_position), np.array(target_position), num_points)
    
    for i, waypoint in enumerate(waypoints):
        try:
            action = inverse_kinematics(ik, robot, *waypoint)
            robot.send_action(action)
        except Exception as e:
            print(f"Error occurred while sending action: {e}")
        
        # Show progress
        if i % 10 == 0:  # Show progress every 10 steps
            progress = (i / num_points) * 100
            print(f"Moving to target position progress: {progress:.1f}%")
        
        time.sleep(0.03)  # Sleep for 20ms between control steps
    
    print("Robot has moved to target position")



def main():
    """Main function"""
    print("LeRobot Simplified Keyboard Control Example (P Control)")
    print("="*50)
    
    try:
        from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
        from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig

        # Configure robot
        print(f"Connecting SO-101 follower on {PORT} (id={ROBOT_ID})")
        robot_config = SOFollowerRobotConfig(port=PORT, id=ROBOT_ID, use_degrees=True)
        robot = SO101Follower(robot_config)
        
        # Configure keyboard
        keyboard_config = KeyboardTeleopConfig()
        keyboard = KeyboardTeleop(keyboard_config)
        
        # Connect devices
        robot.connect()
        keyboard.connect()
        
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
        
        # Move to zero position
        move_to_zero_position(robot, duration=5.0)
        
        
        # Initialize x,y coordinate control (TODO)
        ik = RobotKinematics(URDF, TARGET_FRAME)
        q_now = np.array([robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors])

        pose = ik.forward_kinematics(q_now)
        current_x, current_y, current_z = pose[:3, 3]
        print(f"Initialize end effector position: x={current_x:.4f}, y={current_y:.4f}, z={current_z:.4f}")

        tmp = [TARGET_POSITION[0], TARGET_POSITION[1], TARGET_POSITION[2]+0.05] # move 3cm above the target position first  
        move2target_position(robot, ik, (current_x, current_y, current_z), tmp, duration=2)
        q_now = np.array([robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors])
        pose = ik.forward_kinematics(q_now)
        current_x, current_y, current_z = pose[:3, 3]
        print(f"Current end effector position after moving to above target: x={current_x:.4f}, y={current_y:.4f}, z={current_z:.4f}")
        move2target_position(robot, ik, (current_x, current_y, current_z), TARGET_POSITION, duration=2.5)

        q_now = np.array([robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors])
        pose = ik.forward_kinematics(q_now)
        current_x, current_y, current_z = pose[:3, 3]
        print(f"Current end effector position after moving to target: x={current_x:.4f}, y={current_y:.4f}, z={current_z:.4f}")
        move2target_position(robot, ik, (current_x, current_y, current_z), tmp, duration=1)


        tmp2 = [TARGET_POSITION_2[0], TARGET_POSITION_2[1], TARGET_POSITION_2[2]+0.05] # move 3cm above the target position first
        q_now = np.array([robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors])
        pose = ik.forward_kinematics(q_now)
        current_x, current_y, current_z = pose[:3, 3]
        print(f"Current end effector position after moving back to above target: x={current_x:.4f}, y={current_y:.4f}, z={current_z:.4f}")  
        move2target_position(robot, ik, (current_x, current_y, current_z), tmp2, duration=2.5)

        q_now = np.array([robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors])
        pose = ik.forward_kinematics(q_now)
        current_x, current_y, current_z = pose[:3, 3]
        print(f"Current end effector position after moving to second target: x={current_x:.4f}, y={current_y:.4f}, z={current_z:.4f}")
        move2target_position(robot, ik, (current_x, current_y, current_z), TARGET_POSITION_2, duration=2.5)


        
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
