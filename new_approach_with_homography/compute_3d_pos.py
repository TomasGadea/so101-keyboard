import json
import sys
from pathlib import Path

import numpy as np
import cv2

CALIBRATION_PATH = Path(__file__).parent / "calibration.json"

VERBOSE = False


def vprint(*args, **kwargs):
    """print() that only emits when this module's VERBOSE flag is set."""
    if VERBOSE:
        print(*args, **kwargs)


def load_calibration(path=CALIBRATION_PATH):
    with open(path, "r") as f:
        return json.load(f)


def load_config(path):
    """Load a pipeline config from an explicit path (no default)."""
    with open(path, "r") as f:
        return json.load(f)

class VisionTo3D:
    def __init__(self, calibration_points, plane_z=0.025):
        """
        calibration_points: dict mapping a label to {"pixel": [u, v], "world": [x, y, z]}.
        Pixels are in the same coordinate space the caller will later query with.
        plane_z: height (m) of the workspace plane the homography projects onto.
        """
        # Your new camera position
        # self.camera_pos = np.array([0.26, 0.022, 0.15])
        self.plane_z = plane_z

        self.calibration_points = calibration_points
        self.points_2d = {k: v["pixel"] for k, v in calibration_points.items()}
        self.points_3d = {k: v["world"] for k, v in calibration_points.items()}
        self.homography_matrix = None

    # ==========================================
    # METHOD 1: Homography (Requires 2D Pixels)
    # ==========================================
    def calibrate_homography(self, points_2d_dict):
        """
        Calculates the transformation matrix from the 2D camera view to the 3D plane.
        points_2d_dict should be a dictionary of the pixel coordinates for your 6 points.
        e.g., {'z': [u, v], 'e': [u, v], ...}
        """
        pts_src = [] # 2D pixels
        pts_dst = [] # 3D points (ignoring Z since it's flat)

        for key in self.points_3d.keys():
            if key in points_2d_dict:
                pts_src.append(points_2d_dict[key])
                # We only need X and Y for planar homography
                pts_dst.append([self.points_3d[key][0], self.points_3d[key][1]])

        pts_src = np.array(pts_src, dtype=np.float32)
        pts_dst = np.array(pts_dst, dtype=np.float32)

        # Calculate the homography matrix
        self.homography_matrix, _ = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=0.5)
        vprint("Homography Calibration Complete.")

    def get_3d_from_pixel_homography(self, pixel_u, pixel_v):
        """
        Converts a pixel to a 3D coordinate using the calculated Homography.
        """
        if self.homography_matrix is None:
            raise ValueError("Please run calibrate_homography() first.")

        # Format pixel for matrix multiplication
        pixel_array = np.array([[[pixel_u, pixel_v]]], dtype=np.float32)
        
        # Transform 2D pixel to 2D world coordinate (X, Y)
        world_2d = cv2.perspectiveTransform(pixel_array, self.homography_matrix)
        
        # Add the known Z plane back in
        x, y = world_2d[0][0]
        return np.array([x, y, self.plane_z])


    # ==========================================
    # METHOD 2: Ray Intersect (Requires Cam Matrix)
    # ==========================================
    # def get_3d_from_pixel_ray(self, pixel_u, pixel_v, camera_matrix, rotation_matrix):
    #     """
    #     Converts a pixel to a 3D coordinate by drawing a line from the camera
    #     until it hits the table (Z = -0.03).
    #     Requires intrinsic camera_matrix (K) and camera to world rotation_matrix (R).
    #     """
    #     # 1. Convert pixel to normalized camera coordinates (invert intrinsic matrix)
    #     pixel_vector = np.array([pixel_u, pixel_v, 1.0])
    #     inv_camera_matrix = np.linalg.inv(camera_matrix)
    #     ray_camera = inv_camera_matrix.dot(pixel_vector)
        
    #     # 2. Rotate the ray into the world coordinate frame
    #     ray_world = rotation_matrix.dot(ray_camera)
        
    #     # 3. Find where the ray intersects the Z = -0.03 plane
    #     # Line equation: P = CameraPos + t * Ray
    #     # Since we know P_z = -0.03, we solve for t:
    #     t = (self.plane_z - self.camera_pos[2]) / ray_world[2]
        
    #     # 4. Plug 't' back in to find X and Y
    #     point_3d = self.camera_pos + (t * ray_world)
    #     return point_3d

# ==========================================
# EXAMPLE USAGE
# ==========================================

# Default calibration: 2D pixels (in [0, 1000] normalized coords) paired with
# their corresponding 3D world points (meters).
# DEFAULT_CALIBRATION = {
#     'q':  {'pixel': [133, 395], 'world': [0.2057, 0.1042, 0.025]},
#     'p':  {'pixel': [719, 384], 'world': [0.2087, -0.0414, 0.025]},
#     'z':  {'pixel': [215, 546], 'world': [0.1671, 0.0896, 0.025]},
#     'n':  {'pixel': [543, 533], 'world': [0.1655, 0.0108, 0.025]},
#     'g':  {'pixel': [437, 465], 'world': [0.1794, 0.0382, 0.025]},
#     'u':  {'pixel': [524, 391], 'world': [0.2033, 0.0135, 0.025]},
#     'q2': {'pixel': [228, 310], 'world': [0.2168, 0.1041, 0.025]},
#     'z2': {'pixel': [273, 477], 'world': [0.1710, 0.0869, 0.025]},
#     'z3': {'pixel': [305, 580], 'world': [0.1509, 0.0732, 0.025]},
# }


# Calibration: 2D pixels (in [0, 1000] normalized coords) paired with
# their corresponding 3D world points (meters). Loaded from calibration.json.
DEFAULT_CALIBRATION = load_calibration()

# Default targets in [0, 1000] normalized coordinates
DEFAULT_TARGETS = [(191, 389), (637, 456)]


def run(targets, calibration=DEFAULT_CALIBRATION, verbose=False, *, config):
    """
    Compute 3D coordinates for a list of (target_u, target_v) pixel targets
    given in [0, 1000] normalized coordinates.

    `calibration` is a dict {label: {"pixel": [u, v], "world": [x, y, z]}} where
    pixels are also in [0, 1000] normalized coordinates.
    `config` is the parsed pipeline_config dict; it is required and must be
    passed explicitly by the caller. The workspace plane height (`plane_z`)
    and the camera `image_resolution` are read from it.

    Returns a list of 3D coordinates (np.ndarray of shape (3,)).
    """
    global VERBOSE
    VERBOSE = verbose

    plane_z = config["plane_z"]
    image_width, image_height = config["image_resolution"]

    scaled_calibration = {
        k: {
            "pixel": [v["pixel"][0] / 1000 * image_width,
                      v["pixel"][1] / 1000 * image_height],
            "world": v["world"],
        }
        for k, v in calibration.items()
    }
    vision = VisionTo3D(scaled_calibration, plane_z=plane_z)
    vision.calibrate_homography(vision.points_2d)

    results = []
    for target_u, target_v in targets:
        u = target_u / 1000 * image_width
        v = target_v / 1000 * image_height
        predicted_3d = vision.get_3d_from_pixel_homography(u, v)
        vprint(f"Pixel ({u}, {v}) corresponds to 3D point: X={predicted_3d[0]:.4f}, Y={predicted_3d[1]:.4f}, Z={predicted_3d[2]:.4f}")
        vprint(predicted_3d)
        results.append(predicted_3d)
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <pipeline_config.json>")
    try:
        run(DEFAULT_TARGETS, config=load_config(sys.argv[1]))
    except Exception as e:
        vprint(f"Error testing homography: {e}")