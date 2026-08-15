#!/usr/bin/env bash
# Continuously snap frames from the front camera and overwrite a JPEG file.
# Open the JPEG in VS Code (or any viewer that reloads on change) to get a
# near-live preview while you teleoperate the arm in another terminal.
#
# Usage:
#   bash scripts/preview_camera.sh
# By default writes to ../snap.jpg (visible on the Windows side of WSL2).
# Override:
#   OUT=/some/path.jpg INTERVAL=0.25 bash scripts/preview_camera.sh
set -euo pipefail

CAMERA="${CAMERA:-/dev/video0}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$REPO_DIR/snap.jpg}"
INTERVAL="${INTERVAL:-0.5}"

echo "Preview to $OUT every ${INTERVAL}s. Ctrl+C to stop."
python - "$CAMERA" "$OUT" "$INTERVAL" << 'EOF'
import sys, time, cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation, Cv2Backends

camera_path, out_path, interval = sys.argv[1], sys.argv[2], float(sys.argv[3])
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
    print(f"Camera up. Writing {out_path}")
    while True:
        try:
            frame = cam.async_read(timeout_ms=2000)
            cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
EOF
