#!/usr/bin/env python3
"""
Live camera feed and Kinematics logger for SO100/SO101 robot.
Displays a live OpenCV window. Press 'r' to snap a picture and save it 
using the current End Effector coordinates as the filename.
"""

import cv2
import time
import logging
import numpy as np
import traceback

# Camera imports
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# Robot imports
from lerobot.robots.so_follower.so_follower import SO100Follower
from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig
from lerobot.model.kinematics import RobotKinematics

# --- CONFIGURATION ---
# Camera Config
CAMERA_PATH = 0
# Robot Config
PORT = "/dev/tty.usbmodem5B141136321"
URDF = "/Users/jlafuente/Desktop/robot_learning_project/calibration/so101_new_calib.urdf"
TARGET_FRAME = "gripper_frame_link"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("="*50)
    print("Live Kinematics Snapshot Tool")
    print("="*50)

    # 1. Initialize Robot and Kinematics
    logger.info(f"Connecting to robot on port: {PORT}")
    robot_config = SO100FollowerConfig(port=PORT)
    robot = SO100Follower(robot_config)
    
    try:
        robot.connect()
        logger.info("Robot connected successfully!")

        ik = RobotKinematics(URDF, TARGET_FRAME)

    except Exception as e:
        logger.error(f"Failed to connect to robot: {e}")
        return

    # 2. Initialize Camera
    logger.info(f"Connecting to camera at: {CAMERA_PATH}")
    cam_cfg = OpenCVCameraConfig(
        index_or_path=CAMERA_PATH, 
        fps=15, 
        width=640, 
        height=480,
        color_mode=ColorMode.RGB, 
        rotation=Cv2Rotation.NO_ROTATION,
        fourcc="MJPG", 
        warmup_s=2
    )
    
    try:
        cam = OpenCVCamera(cam_cfg)
        cam.connect()
        logger.info("Camera connected successfully!")
    except Exception as e:
        logger.error(f"Failed to connect to camera: {e}")
        robot.disconnect()
        return

    # 3. Main Loop
    print("\n" + "="*50)
    print("INSTRUCTIONS:")
    print(" - Make sure the 'Camera Feed' window is active/selected.")
    print(" - Press 'r' to save a picture with current coordinates.")
    print(" - Press 'q' to quit the program.")
    print("="*50 + "\n")

    try:
        while True:
            # Grab latest frame
            frame_rgb = cam.async_read(timeout_ms=2000)
            
            # Convert to BGR for OpenCV display and saving
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Show the live feed
            cv2.imshow("Camera Feed", frame_bgr)
            
            # Wait 1ms for keypress
            key = cv2.waitKey(1) & 0xFF
            
            # If 'r' is pressed: Record
            if key == ord('r'):
                # Read current joint angles
                q_now = np.array([robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors])
                
                # Calculate Forward Kinematics
                pose = ik.forward_kinematics(q_now)
                current_x, current_y, current_z = pose[:3, 3]
                
                # Format filename: x_y_z.jpg
                filename = f"coord_x{current_x:.4f}_y{current_y:.4f}_z{current_z:.4f}.jpg"
                
                # Save the image
                cv2.imwrite(filename, frame_bgr)
                logger.info(f"SNAP! Saved image: {filename}")

            # If 'q' is pressed: Quit
            elif key == ord('q'):
                logger.info("Quitting program...")
                break

    except Exception as e:
        logger.error(f"Error during runtime: {e}")
        traceback.print_exc()

    finally:
        # Cleanup
        logger.info("Disconnecting devices...")
        cam.disconnect()
        cv2.destroyAllWindows()
        robot.disconnect()
        logger.info("Done.")

if __name__ == "__main__":
    main()