#!/usr/bin/env bash
# Save one frame from the front camera to a file you can open with any viewer.
# Use this before recording to verify the camera angle without launching rerun.
#
# Usage:
#   bash scripts/snap_camera.sh           # saves to /tmp/snap.jpg
#   OUT=/some/path.jpg bash scripts/snap_camera.sh
set -euo pipefail

CAMERA="${CAMERA:-/dev/video0}"
OUT="${OUT:-snap.jpg}"

# Use lerobot's OpenCVCamera (with MJPG) so this matches what record_pilot.sh sees.
python - "$CAMERA" "$OUT" << 'EOF'
import sys, cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation, Cv2Backends

camera_path, out_path = sys.argv[1], sys.argv[2]
if isinstance(camera_path, str):
    if camera_path.startswith("opencv:"):
        camera_path = camera_path.split(":", 1)[1]
    if camera_path.isdigit():
        camera_path = int(camera_path)
cfg = OpenCVCameraConfig(
    index_or_path=camera_path, fps=30, width=640, height=480,
    color_mode=ColorMode.RGB, rotation=Cv2Rotation.NO_ROTATION,
    fourcc="MJPG", warmup_s=2,
    backend=Cv2Backends.V4L2)
with OpenCVCamera(cfg) as cam:
    # Discard a couple of warmup frames, then grab one.
    for _ in range(3):
        cam.async_read(timeout_ms=2000)
    frame_rgb = cam.async_read(timeout_ms=2000)
cv2.imwrite(out_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
print(f"saved {out_path} ({frame_rgb.shape})")
EOF
