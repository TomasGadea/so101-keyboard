Installation
This guide uses conda (via miniforge) to manage environments (recommended). If you prefer another environment manager (e.g. uv, venv), ensure you have Python >=3.12 and support PyTorch >= 2.10, then skip ahead to Environment Setup.

Step 1 ( conda only): Install miniforge
Copied
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
Step 2: Environment Setup
Create a virtual environment with Python 3.12:

Copied
conda create -y -n lerobot python=3.12
Then activate your virtual environment, you have to do this each time you open a shell to use lerobot:

Copied
conda activate lerobot
When installing LeRobot inside WSL (Windows Subsystem for Linux), make sure to also install evdev:

Copied
conda install evdev -c conda-forge
Install ffmpeg (for video decoding)
LeRobot uses TorchCodec for video decoding by default, which requires ffmpeg.

Platform support: TorchCodec is not available on macOS Intel (x86_64), Linux ARM (aarch64, arm64, armv7l), or Windows with PyTorch < 2.8. On these platforms, LeRobot automatically falls back to pyav — so you do not need to install ffmpeg and can skip to Step 3.

If your platform supports TorchCodec, install ffmpeg using one of the methods below:

Install ffmpeg in your conda environment. This works with all PyTorch versions and is required for PyTorch < 2.10:

Copied
conda install ffmpeg -c conda-forge
This usually installs ffmpeg 8.X with the libsvtav1 encoder. If you run into issues (e.g. libsvtav1 missing — check with ffmpeg -encoders — or a version mismatch with torchcodec), you can explicitly install ffmpeg 7.1.1 using:

Copied
conda install ffmpeg=7.1.1 -c conda-forge
Step 3: Install LeRobot 🤗
From Source
First, clone the repository and navigate into the directory:

Copied
git clone https://github.com/huggingface/lerobot.git
cd lerobot
Then, install the library in editable mode. This is useful if you plan to contribute to the code.

Copied
pip install -e .
Installation from PyPI
Core Library: Install the base package with:

Copied
pip install lerobot
This installs only the default dependencies.

Extra Features: To install additional functionality, use one of the following (If you are using uv, replace pip install with uv pip install in the commands below.):

Copied
pip install 'lerobot[all]'          # All available features
pip install 'lerobot[aloha,pusht]'  # Specific features (Aloha & Pusht)
pip install 'lerobot[feetech]'      # Feetech motor support
Replace [...] with your desired features.

Available Tags: For a full list of optional dependencies, see: https://pypi.org/project/lerobot/

Troubleshooting
If you encounter build errors, you may need to install additional dependencies: cmake, build-essential, and ffmpeg libs. To install these for Linux run:

Copied
sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
For other systems, see: Compiling PyAV

Optional dependencies
LeRobot provides optional extras for specific functionalities. Multiple extras can be combined (e.g., .[aloha,feetech]). For all available extras, refer to pyproject.toml. If you are using uv, replace pip install with uv pip install in the commands below.

Simulations
Install environment packages: aloha (gym-aloha), or pusht (gym-pusht) Example:

Copied
pip install -e ".[aloha]" # or "[pusht]" for example
Motor Control
For Koch v1.1 install the Dynamixel SDK, for SO100/SO101/Moss install the Feetech SDK.

Copied
pip install -e ".[feetech]" # or "[dynamixel]" for example
Experiment Tracking
To use Weights and Biases for experiment tracking, log in with

Copied
wandb login
You can now assemble your robot if it’s not ready yet, look for your robot type on the left. Then follow the link below to use Lerobot with your robot.

Update on GitHub
←