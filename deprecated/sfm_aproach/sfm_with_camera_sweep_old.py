import pycolmap
from pathlib import Path
import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import cv2
import io
import os
import json

from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types


from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

from lerobot.robots.so_follower.so_follower import SO100Follower
from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig
from lerobot.model.kinematics import RobotKinematics
        

# Paths for sfm
project_path = Path("/Users/jlafuente/Desktop/robot_learning_project")
image_path = project_path / "images"
output_path = project_path / "sparse"

PORT = "/dev/tty.usbmodem5B141125291"
CAMERA_PATH = 0
ROBOT_ID = "keyboard_follower"
URDF = "/Users/jlafuente/Desktop/robot_learning_project/calibration/so101_new_calib.urdf"
TARGET_FRAME = "gripper_frame_link"

API_KEY = "...."
KEYS_JSON = "keys.json"
TARGET_SIZE = (640, 480)
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


JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],      # Joint 1: zero position offset, scale factor
    ['shoulder_lift', 2.0, 0.97],     # Joint 2: zero position offset, scale factor
    ['elbow_flex', 0.0, 1.05],        # Joint 3: zero position offset, scale factor
    ['wrist_flex', 0.0, 0.94],        # Joint 4: zero position offset, scale factor
    ['wrist_roll', 0.0, 0.5],        # Joint 5: zero position offset, scale factor
    ['gripper', 0.0, 1.0],           # Joint 6: zero position offset, scale factor
]


MODEL = "models/gemini-3-flash-preview"

def image_to_part(image: Image.Image) -> types.Part:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


def call_vlm(client: genai.Client, prompt: str, image_part: types.Part, label: str = "vlm") -> dict:
    start = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, image_part],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    elapsed = time.perf_counter() - start
    print(f"[{label}] Google API call took {elapsed:.2f}s")
    return json.loads(response.text)


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

def compute_sim3_calibration(colmap_points, robot_points):
    """
    Computes Scale, Rotation, and Translation to align COLMAP points to Robot points.
    Uses the Umeyama algorithm.
    """
    A = np.array(colmap_points)
    B = np.array(robot_points)
    
    assert len(A) >= 3, "Need at least 3 points for a full 3D transform"
    assert A.shape == B.shape, "Point arrays must have the same shape"
    
    # 1. Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # 2. Center the points
    AA = A - centroid_A
    BB = B - centroid_B
    
    # 3. Variance of A
    var_A = np.mean(np.sum(AA**2, axis=1))
    
    # 4. Compute covariance matrix H
    H = AA.T @ BB / A.shape[0]
    
    # 5. SVD to find Rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case (ensure it's a valid rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        S[-1] *= -1
        
    # 6. Compute exact Scale
    scale = np.sum(S) / var_A
    
    # 7. Compute exact Translation
    t = centroid_B - scale * (R @ centroid_A)
    
    return scale, R, t


def take_picture(cam, ik, path_images):
    frame_rgb = cam.async_read(timeout_ms=2000)
            
    # Convert to BGR for OpenCV display and saving
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Show the live feed
    cv2.imshow("Camera Feed", frame_bgr)

    # Read current joint angles
    q_now = np.array([robot.get_observation()[f"{m}.pos"] for m in robot.bus.motors])
    
    # Calculate Forward Kinematics (TODO: get the actual position not just the result of FK)
    pose = ik.forward_kinematics(q_now)
    current_x, current_y, current_z = pose[:3, 3]
    
    # Format filename: x_y_z.jpg
    filename = f"coord_x{current_x:.4f}_y{current_y:.4f}_z{current_z:.4f}.jpg"
    
    # Save the image
    cv2.imwrite(path_images / filename, frame_bgr)

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
    print("Using P control to slowly move robot to zero position...")
    
    # Get current robot state
    current_obs = robot.get_observation()
    
    # Extract current joint positions
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            current_positions[motor_name] = value
    
    
    # Calculate control steps
    control_freq = 50  # 50Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq
    
    print(f"Will use P control to move to specified positions in {duration} seconds, control frequency: {control_freq}Hz, proportional gain: {kp}")
    
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
            print(f"Moving to specified positions progress: {progress:.1f}%")
        
        time.sleep(step_time)
    
    print("Robot has moved to specified positions")


def get_ray_from_pixel(reconstruction, image_id, pixel_x, pixel_y): 
    image = reconstruction.images[image_id]
    camera = reconstruction.cameras[image.camera_id]
    
    # 1. Origin: Camera Center in World Coordinates
    ray_origin = image.projection_center() 
    
    # 2. Pixel -> Camera Space (Normalized)
    pixel_cam = camera.cam_from_img(np.array([pixel_x, pixel_y]))
    direction_camera = np.array([pixel_cam[0], pixel_cam[1], 1.0])
    
    # 3. Rotation: Camera-to-World
    pose = image.cam_from_world() 
    try:
        r_w2c = pose.rotation.matrix()
    except AttributeError:
        q = pose.rotation.quat
        scipy_quat = [q[1], q[2], q[3], q[0]] # Convert to [x, y, z, w]
        r_w2c = R.from_quat(scipy_quat).as_matrix()

    r_c2w = r_w2c.T
    
    # 4. Transform Direction to World Space
    ray_direction = r_c2w @ direction_camera
    ray_direction /= np.linalg.norm(ray_direction) # Normalize to unit vector
    
    return ray_origin, ray_direction

def triangulate_two_rays(o1, d1, o2, d2):
    """
    Finds the closest point of intersection between two 3D rays.
    Returns the 3D coordinate (XYZ).
    """
    w0 = o1 - o2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)

    denominator = a * c - b * b
    if denominator < 1e-8:
        raise ValueError("Rays are parallel, cannot triangulate.")

    # Calculate distance along each ray to the closest point
    s = (b * e - c * d) / denominator
    t = (a * e - b * d) / denominator

    # Get the 3D points on each ray where they are closest
    p1 = o1 + s * d1
    p2 = o2 + t * d2

    # The actual 3D point is the midpoint of the shortest connecting segment
    intersection = (p1 + p2) / 2.0
    return intersection

def get_coords_from_filename(filename):
    # Extracts numbers following x, y, and z in the filename
    matches = re.findall(r'[xyz]([-+]?\d*\.\d+|\d+)', filename)
    return np.array([float(m) for m in matches])

def apply_sim3_transform(colmap_point, scale, rotation_matrix, translation_vector):
    """
    Transforms an unaligned COLMAP point into an exact metric Robot coordinate space.
    
    Args:
        colmap_point (np.ndarray): The [X, Y, Z] point from COLMAP (shape: 3,)
        scale (float): The scaling factor to convert COLMAP units to Robot units (e.g., meters)
        rotation_matrix (np.ndarray): 3x3 rotation matrix aligning COLMAP to the Robot
        translation_vector (np.ndarray): 3x1 or (3,) vector representing the origin offset
        
    Returns:
        np.ndarray: The [X, Y, Z] point in the exact Robot coordinate system.
    """
    colmap_point = np.array(colmap_point)
    rotation_matrix = np.array(rotation_matrix)
    translation_vector = np.array(translation_vector)
    
    # 1. Rotate the point to match the robot's axis orientation
    rotated_point = rotation_matrix @ colmap_point
    
    # 2. Scale the point to match the robot's metric units
    scaled_point = scale * rotated_point
    
    # 3. Translate the point to anchor it to the robot's physical origin
    robot_point = scaled_point + translation_vector
    
    return robot_point

if __name__ == "__main__":
    output_path.mkdir(exist_ok=True)
    # Ensure the project directory exists
    project_path.mkdir(parents=True, exist_ok=True)

    # robot_config = SO100FollowerConfig(port=PORT)
    # robot = SO100Follower(robot_config)
    # robot.connect()

    # ik = RobotKinematics(URDF, TARGET_FRAME)

    # cam_cfg = OpenCVCameraConfig(
    #     index_or_path=CAMERA_PATH, 
    #     fps=15, 
    #     width=640, 
    #     height=480,
    #     color_mode=ColorMode.RGB, 
    #     rotation=Cv2Rotation.NO_ROTATION,
    #     fourcc="MJPG", 
    #     warmup_s=2
    # )
    # cam = OpenCVCamera(cam_cfg)
    # cam.connect()

    # while True:
    #     calibrate_choice = input("Do you want to recalibrate the robot? (y/n): ").strip().lower()
    #     if calibrate_choice in ['y', 'yes']:
    #         print("Starting recalibration...")
    #         robot.calibrate()
    #         print("Calibration completed!")
    #         break
    #     elif calibrate_choice in ['n', 'no']:
    #         print("Using previous calibration file")
    #         break
    #     else:
    #         print("Please enter y or n")

    # default_positions = {
    #     'shoulder_pan': 0.0,
    #     'shoulder_lift': -35.0,
    #     'elbow_flex': 0.0,
    #     'wrist_flex': 100.0,
    #     'wrist_roll': -135.0,
    #     'gripper': 80.0
    # }

    # move_to_joint_position(robot, joint_positions=default_positions, duration=1.0, kp=0.5)

    # # Reading joint positions
    # print("Reading initial joint angles...")
    # start_obs = robot.get_observation()
    # start_positions = {}
    # for key, value in start_obs.items():
    #     if key.endswith('.pos'):
    #         motor_name = key.removesuffix('.pos')
    #         start_positions[motor_name] = int(value)
    #         print(f"{motor_name}: {value}")

    # # Sweep through a range of positions in the first joint to take pictures
    # for pos in [-15, 0, 15]:
    #     print(f"\nMoving shoulder_pan to {pos} degrees...")
    #     target_positions = default_positions.copy()
    #     target_positions['shoulder_pan'] = pos
    #     move_to_joint_position(robot, joint_positions=target_positions, duration=0.5, kp=0.5)
        
    #     print("Taking picture at current position...")
    #     time.sleep(0.1)  
    #     take_picture(cam, ik, image_path)
    #     time.sleep(0.1)  # Short delay before next movement


    # zero_positions = {
    #     'shoulder_pan': 0.0,
    #     'shoulder_lift': 0.0,
    #     'elbow_flex': 0.0,
    #     'wrist_flex': 60.0,
    #     'wrist_roll': -135.0,
    #     'gripper': 80.0
    # }

    # move_to_joint_position(robot, joint_positions=zero_positions, duration=1.0, kp=0.5)

    # 1. Feature Extraction
    # This populates the 'images' and 'keypoints' tables in the DB
    # Create ImageReaderOptions to specify the camera model
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "SIMPLE_RADIAL"

    # If you want to use a single camera for ALL images (highly recommended for calibration)
    # Options: AUTO, INDIVIDUAL, SINGLE
    camera_mode = pycolmap.CameraMode.AUTO

    print("Extracting features...")
    pycolmap.extract_features(
        database_path=project_path / "database_2.db",
        image_path=image_path,
        camera_mode=camera_mode,
        reader_options=reader_options
    )

    print("Matching features...")
    pycolmap.match_exhaustive(database_path=project_path / "database_2.db")

    # Run the full reconstruction pipeline
    reconstruction = pycolmap.incremental_mapping(
        database_path=project_path / "database_2.db",
        image_path=image_path,
        output_path=output_path
    )[0]

    # Prepare alignment data
    image_names = []
    ref_locations = []

    for image_id, image in reconstruction.images.items():
        image_names.append(image.name)
        ref_locations.append(get_coords_from_filename(image.name))

    # Perform Robust Alignment (Sim3 transformation: Scale, Rotation, Translation)
    ransac_options = pycolmap.RANSACOptions()

    # Image 1 is the leftmost image, Image 2 is the rightmost image (based on the order of taking pictures in the sweep)
    all_images = sorted(os.listdir(image_path), key=lambda x: float(x.split('_')[1].split('x')[1]))
    image_path_1 = image_path / all_images[0]
    image_path_2 = image_path / all_images[-1]

    image_1 = Image.open(image_path_1).convert("RGB") 
    image_2 = Image.open(image_path_2).convert("RGB")
    image_1 = image_1.resize(TARGET_SIZE, Image.LANCZOS)
    image_2 = image_2.resize(TARGET_SIZE, Image.LANCZOS)

    # client = genai.Client(api_key=API_KEY)
    # image_1_part = image_to_part(image_1)
    # image_2_part = image_to_part(image_2)
    # letters_data_1 = call_vlm(client, LETTERS_PROMPT, image_1_part, label="letters")
    # letters_data_2 = call_vlm(client, LETTERS_PROMPT, image_2_part, label="letters")

    target_image_1 = image_path_1
    correct_origin_1 = ref_locations[0]  

    # target_pixel_1 = letters_data_1['letters'][0] # Replace with actual letter key you want to target, e.g., 'A'

    # print(f"Target pixel in Image 1: {target_pixel_1}")

    target_image_2 = image_path_2
    correct_origin_2 = ref_locations[-1]    
    # target_pixel_2 = letters_data_2['letters'][0] # Replace with actual letter key you want to target, e.g., 'A'

    # print(f"Target pixel in Image 2: {target_pixel_2}") 

    # target_pixel_1 = (target_pixel_1['x']/1000 * TARGET_SIZE[0], target_pixel_1['y']/1000 * TARGET_SIZE[1]) # Convert to pixel coordinates
    # target_pixel_2 = (target_pixel_2['x']/1000 * TARGET_SIZE[0], target_pixel_2['y']/1000 * TARGET_SIZE[1]) # Convert to pixel coordinates

    # print(f"Target pixel in Image 1: {target_pixel_1}")
    # print(f"Target pixel in Image 2: {target_pixel_2}")

    # Target pixel in Image 1: (294.40000000000003, 97.91999999999999)
    # Target pixel in Image 2: (92.8, 160.32000000000002)

    target_pixel_1 = (294.4, 97.9)
    target_pixel_2 = (92.8, 160.3)

    # target_pixel_1 = (378, 183) # Center of the image for testing
    # target_pixel_1 = (target_pixel_1[0] / TARGET_SIZE[0] * 1000, target_pixel_1[1] / TARGET_SIZE[1] * 1000) # Convert to normalized coordinates
    # target_pixel_2 = (51, 401) # Center of the image for testing
    # target_pixel_2 = (target_pixel_2[0] / TARGET_SIZE[0] * 1000, target_pixel_2[1] / TARGET_SIZE[1] * 1000) # Convert to normalized coordinates


    # 2. Look up the internal COLMAP IDs for these images
    id_lookup = {img.name: id for id, img in reconstruction.images.items()}
    img_id_1 = id_lookup[target_image_1.name.split('/')[-1]] # Extract filename from path for lookup
    img_id_2 = id_lookup[target_image_2.name.split('/')[-1]] # Extract filename from path for lookup
    img_id_3 = id_lookup[all_images[len(all_images)//2]] # Middle image as a reference point for testing
    correct_origin_3 = ref_locations[len(all_images)//2] # Reference point from the middle image for testing

    print(f"Image 1 ID: {img_id_1}, Image 2 ID: {img_id_2}, Image 3 ID: {img_id_3}")

    # 3. Get the Rays (Origin and Direction) for both images
    o1, d1 = get_ray_from_pixel(reconstruction, img_id_1, target_pixel_1[0], target_pixel_1[1])
    o2, d2 = get_ray_from_pixel(reconstruction, img_id_2, target_pixel_2[0], target_pixel_2[1])
    o3 = reconstruction.images[img_id_3].projection_center()

    # 4. Intersect the rays to find the 3D Point
    final_3d_position = triangulate_two_rays(o1, d1, o2, d2)

    print(f"--- Triangulation Results ---")
    print(f"Ray 1 Origin: {o1}")
    print(f"Ray 2 Origin: {o2}")
    print(f"Calculated 3D End Position: {final_3d_position}")

    colamp_points = [o1, o2, o3]
    colmap_points = [(o[2], o[0], o[1]) for o in colamp_points] # Extract XYZ from projection center

    scale_calib, R_calib, t_calib = compute_sim3_calibration(colmap_points, [correct_origin_1, correct_origin_2, correct_origin_3])

    robot_target_position = apply_sim3_transform(
        colmap_point=final_3d_position,
        scale=scale_calib,
        rotation_matrix=R_calib,
        translation_vector=t_calib
    )



    # delete database_2.db to clean up
    db_path = project_path / "database_2.db"
    if db_path.exists():
        db_path.unlink()
        print("Cleaned up database_2.db")

