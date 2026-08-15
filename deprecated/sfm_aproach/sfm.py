import pycolmap
from pathlib import Path
import re
import numpy as np
from scipy.spatial.transform import Rotation as R


project_path = Path("/Users/jlafuente/Desktop/robot_learning_project")
image_path = project_path / "images"
output_path = project_path / "sparse"

output_path.mkdir(exist_ok=True)
# Ensure the project directory exists
project_path.mkdir(parents=True, exist_ok=True)

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
    database_path=project_path / "database.db",
    image_path=image_path,
    camera_mode=camera_mode,
    reader_options=reader_options
)

print("Matching features...")
pycolmap.match_exhaustive(database_path=project_path / "database.db")

# Run the full reconstruction pipeline
reconstruction = pycolmap.incremental_mapping(
    database_path=project_path / "database.db",
    image_path=image_path,
    output_path=output_path
)[0]

# for image_id, image in reconstruction.images.items():
#     print(f"Image: {image.name}, Position: {image.projection_center()}")


def get_coords_from_filename(filename):
    # Extracts numbers following x, y, and z in the filename
    matches = re.findall(r'[xyz]([-+]?\d*\.\d+|\d+)', filename)
    return np.array([float(m) for m in matches])

# Prepare alignment data
image_names = []
ref_locations = []

for image_id, image in reconstruction.images.items():
    image_names.append(image.name)
    ref_locations.append(get_coords_from_filename(image.name))

# Perform Robust Alignment (Sim3 transformation: Scale, Rotation, Translation)
ransac_options = pycolmap.RANSACOptions()

pycolmap.align_reconstruction_to_locations(
    reconstruction,
    image_names,
    ref_locations,
    min_common_images=3, # Changed from points to images
    ransac_options=ransac_options
)


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


# --- USAGE: TRIANGULATING THE 3D POINT ---

# 1. Define your two images and the 2D target pixels in each
target_image_1 = "coord_x0.2060_y0.0226_z0.2277.jpg"
target_pixel_1 = (960, 540) # Replace with actual (x, y) for image 1

target_image_2 = "coord_x0.2231_y0.0105_z0.1768.jpg"
target_pixel_2 = (980, 560) # Replace with actual (x, y) for image 2

# 2. Look up the internal COLMAP IDs for these images
id_lookup = {img.name: id for id, img in reconstruction.images.items()}
img_id_1 = id_lookup[target_image_1]
img_id_2 = id_lookup[target_image_2]

# 3. Get the Rays (Origin and Direction) for both images
o1, d1 = get_ray_from_pixel(reconstruction, img_id_1, target_pixel_1[0], target_pixel_1[1])
o2, d2 = get_ray_from_pixel(reconstruction, img_id_2, target_pixel_2[0], target_pixel_2[1])

# 4. Intersect the rays to find the 3D Point
final_3d_position = triangulate_two_rays(o1, d1, o2, d2)

print(f"--- Triangulation Results ---")
print(f"Ray 1 Origin: {o1}")
print(f"Ray 2 Origin: {o2}")
print(f"Calculated 3D End Position: {final_3d_position}")
