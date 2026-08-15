import matplotlib.pyplot as plt

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from dotenv import load_dotenv
import os

load_dotenv()
CAMERA_INDEX = int(os.environ["CAMERA_INDEX"])

def main():
    config = OpenCVCameraConfig(
        index_or_path=CAMERA_INDEX,
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
    )

    with OpenCVCamera(config) as camera:
        print("Close the window to quit.")

        plt.ion()
        fig, ax = plt.subplots()
        ax.set_axis_off()
        fig.tight_layout()

        frame = camera.read()
        im = ax.imshow(frame)

        while plt.fignum_exists(fig.number):
            try:
                frame = camera.async_read(timeout_ms=500)
            except TimeoutError:
                continue

            im.set_data(frame)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        plt.ioff()


if __name__ == "__main__":
    main()