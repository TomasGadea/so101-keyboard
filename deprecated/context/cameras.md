Cameras
LeRobot offers multiple options for video capture:

Class	Supported Cameras
OpenCVCamera	Phone, built-in laptop, USB webcams
ZMQCamera	Network-connected cameras
RealSenseCamera	Intel RealSense (with depth)
Reachy2Camera	Reachy 2 robot cameras
For OpenCVCamera compatibility details, see the Video I/O with OpenCV Overview.

Find your camera
Every camera requires a unique identifier to be instantiated, allowing you to distinguish between multiple connected devices.

OpenCVCamera and RealSenseCamera support auto-discovery. Run the command below to list available devices and their identifiers. Note that these identifiers may change after rebooting your computer or re-plugging the camera, depending on your operating system.

Copied
lerobot-find-cameras opencv # or realsense for Intel Realsense cameras
The output will look something like this if you have two cameras connected:

Copied
--- Detected Cameras ---
Camera #0:
  Name: OpenCV Camera @ 0
  Type: OpenCV
  Id: 0
  Backend api: AVFOUNDATION
  Default stream profile:
    Format: 16.0
    Width: 1920
    Height: 1080
    Fps: 15.0
--------------------
(more cameras ...)
When using Intel RealSense cameras in macOS, you could get this error: Error finding RealSense cameras: failed to set power state, this can be solved by running the same command with sudo permissions. Note that using RealSense cameras in macOS is unstable.

ZMQCamera and Reachy2Camera do not support auto-discovery. They must be configured manually by providing their network address and port or robot SDK settings.

Use cameras
Frame access modes
All camera classes implement three access modes for capturing frames:

Method	Behavior	Blocks?	Best For
read()	Waits for the camera hardware to return a frame. May block for a long time depending on the camera and SDK.	Yes	Simple scripts, sequential capture
async_read(timeout_ms)	Returns the latest unconsumed frame from background thread. Blocks only if buffer is empty, up to timeout_ms. Raises TimeoutError if no frame arrives.	With a timeout	Control loops synchronized to camera FPS
read_latest(max_age_ms)	Peeks at the most recent frame in buffer (may be stale). Raises TimeoutError if frame is older than max_age_ms.	No	UI visualization, logging, monitoring
Usage examples
The following examples show how to use the camera API to configure and capture frames from different camera types.

Blocking and non-blocking frame capture using an OpenCV-based camera
Color and depth capture using an Intel RealSense camera
Failing to cleanly disconnect cameras can cause resource leaks. Use the context manager protocol to ensure automatic cleanup:

Copied
with OpenCVCamera(config) as camera:
    ...
You can also call connect() and disconnect() manually, but always use a finally block for the latter.

Copied
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# Construct an `OpenCVCameraConfig` with your desired FPS, resolution, color mode, and rotation.
config = OpenCVCameraConfig(
    index_or_path=0,
    fps=15,
    width=1920,
    height=1080,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)

# Instantiate and connect an `OpenCVCamera`, performing a warm-up read (default).
with OpenCVCamera(config) as camera:

    # Read a frame synchronously — blocks until hardware delivers a new frame
    frame = camera.read()
    print(f"read() call returned frame with shape:", frame.shape)

    # Read a frame asynchronously with a timeout — returns the latest unconsumed frame or waits up to timeout_ms for a new one
    try:
        for i in range(10):
            frame = camera.async_read(timeout_ms=200)
            print(f"async_read call returned frame {i} with shape:", frame.shape)
    except TimeoutError as e:
        print(f"No frame received within timeout: {e}")

    # Instantly return a frame - returns the most recent frame captured by the camera
    try:
        initial_frame = camera.read_latest(max_age_ms=1000)
        for i in range(10):
            frame = camera.read_latest(max_age_ms=1000)
            print(f"read_latest call returned frame {i} with shape:", frame.shape)
            print(f"Was a new frame received by the camera? {not (initial_frame == frame).any()}")
    except TimeoutError as e:
        print(f"Frame too old: {e}")
Use your phone’s camera
To use your iPhone as a camera on macOS, enable the Continuity Camera feature:

Ensure your Mac is running macOS 13 or later, and your iPhone is on iOS 16 or later.
Sign in both devices with the same Apple ID.
Connect your devices with a USB cable or turn on Wi-Fi and Bluetooth for a wireless connection.
For more details, visit Apple support.

If everything is set up correctly, your phone will appear as a standard OpenCV camera and can be used with OpenCVCamera.

Update on GitHub