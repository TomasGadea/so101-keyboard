"""
Identify the camera on /dev/videoN and (optionally) calibrate it.

How to run:
    # 1) Print device info (driver, card name, supported formats):
    python calibrate_camera.py --info

    # 2) Capture checkerboard images interactively. Print a checkerboard
    #    (default 9x6 inner corners, e.g. https://markhedleyjones.com/projects/calibration-checkerboard-collection),
    #    hold it in front of the camera, press SPACE to save a frame, ESC to quit.
    #    Aim for 15-25 frames covering different angles, distances, and corners of the FOV.
    python calibrate_camera.py --capture --out calib_images

    # 3) Run calibration on the captured images and write intrinsics.json:
    python calibrate_camera.py --calibrate --images calib_images --square-size 0.025

Outputs intrinsics.json with cameraMatrix (K), distCoeffs, image_size, and
the reprojection error. If reprojection error < ~0.5 px, calibration is good.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np

CAMERA_INDEX = 4
WIDTH, HEIGHT = 640, 480
FPS = 30


def cmd_info(index: int) -> None:
    device = f"/dev/video{index}"
    print(f"Querying {device} ...")
    try:
        out = subprocess.run(
            ["v4l2-ctl", "--device", device, "--all"],
            capture_output=True, text=True, check=False,
        )
        print(out.stdout or out.stderr)
    except FileNotFoundError:
        print("v4l2-ctl not installed. Try: sudo apt install v4l-utils")

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Could not open {device} via OpenCV.")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"OpenCV reports: {w}x{h} @ {fps:.1f} fps")
    cap.release()


def cmd_capture(index: int, out_dir: str, board_size) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        print(f"Could not open /dev/video{index}")
        sys.exit(1)

    saved = len(glob.glob(os.path.join(out_dir, "frame_*.png")))
    print("SPACE: save frame   ESC: quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, board_size, corners, found)
        cv2.putText(view, f"saved: {saved}  found: {found}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("calibrate", view)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == 32 and found:
            path = os.path.join(out_dir, f"frame_{saved:03d}.png")
            cv2.imwrite(path, frame)
            saved += 1
            print(f"saved {path}")
    cap.release()
    cv2.destroyAllWindows()


def cmd_calibrate(images_dir: str, board_size, square_size: float,
                  out_json: str) -> None:
    cols, rows = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    obj_points, img_points = [], []
    image_size = None
    paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not paths:
        print(f"No PNG images in {images_dir}")
        sys.exit(1)

    for p in paths:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not found:
            print(f"  no corners: {p}")
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
        )
        obj_points.append(objp)
        img_points.append(corners)
        print(f"  ok: {p}")

    if len(obj_points) < 5:
        print(f"Not enough valid frames ({len(obj_points)}); need >=5.")
        sys.exit(1)

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None,
    )

    result = {
        "image_size": list(image_size),
        "cameraMatrix": K.tolist(),
        "distCoeffs": dist.ravel().tolist(),
        "reprojection_error_px": float(rms),
        "num_frames": len(obj_points),
        "checkerboard_inner_corners": list(board_size),
        "square_size_m": square_size,
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_json}")
    print(f"reprojection error: {rms:.3f} px  (good: <0.5, ok: <1.0)")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
    print(f"distCoeffs (k1,k2,p1,p2,k3,...): {result['distCoeffs']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=CAMERA_INDEX)
    p.add_argument("--info", action="store_true")
    p.add_argument("--capture", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--out", default="calib_images")
    p.add_argument("--images", default="calib_images")
    p.add_argument("--board", default="9x6",
                   help="inner corners as COLSxROWS (default 9x6)")
    p.add_argument("--square-size", type=float, default=0.025,
                   help="checkerboard square edge length in meters")
    p.add_argument("--out-json", default="intrinsics.json")
    args = p.parse_args()

    cols, rows = (int(x) for x in args.board.lower().split("x"))
    board_size = (cols, rows)

    if args.info:
        cmd_info(args.index)
    if args.capture:
        cmd_capture(args.index, args.out, board_size)
    if args.calibrate:
        cmd_calibrate(args.images, board_size, args.square_size, args.out_json)
    if not (args.info or args.capture or args.calibrate):
        p.print_help()


if __name__ == "__main__":
    main()
