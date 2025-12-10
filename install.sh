yes | /home/user/myenv/bin/pip uninstall opencv-python
yes | /home/user/myenv/bin/pip uninstall numpy -y

# Enhanced security and configuration
# Use environment variables for user paths and avoid hardcoded credentials
set -e

# Set default user and project paths from environment or fallback
USER_HOME="${USER_HOME:-/home/user}"
VENV_NAME="${VENV_NAME:-myenv}"
PROJECT_DIR="${PROJECT_DIR:-$USER_HOME/football-ai/football-project-v1}"
MODELS_DIR="$PROJECT_DIR/models"
INPUTS_DIR="$PROJECT_DIR/inputs"

# Create virtual environment
python3 -m venv "$USER_HOME/$VENV_NAME"
source "$USER_HOME/$VENV_NAME/bin/activate"

# Install dependencies
pip install --upgrade pip
pip install -r "$PROJECT_DIR/requirements.txt"

# Remove opencv-python if present
yes | pip uninstall opencv-python

# Install gdown if missing
if ! command -v gdown &> /dev/null; then
    pip install gdown
fi

# Download models
mkdir -p "$MODELS_DIR"
gdown -O "$MODELS_DIR/yolov8x-football.pt" "https://drive.google.com/uc?id=1JS8hpKHkERWGEsIQK7UPCg16O_5IFRmN"
gdown -O "$MODELS_DIR/key_points_pitch_ver2.pt" "https://drive.google.com/uc?id=1ul_FCU03J2PYiup-WTgcW5YcUHlDrmLY"

# Download input video
mkdir -p "$INPUTS_DIR"
gdown -O "$INPUTS_DIR/ok_798b45_0.mp4" "https://drive.google.com/uc?id=1RDTNqZMxbZIOYGRaBaUYz6R2d07QG3vH"

# Rebuild OpenCV (optional, only if needed)
# sudo apt install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev
# cd "$USER_HOME"
# git clone https://github.com/opencv/opencv.git
# git clone https://github.com/opencv/opencv_contrib.git
# cd opencv
# mkdir build && cd build
# cmake -D WITH_FFMPEG=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D PYTHON3_EXECUTABLE=$(which python3) -D PYTHON3_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") -D PYTHON3_PACKAGES_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])") ..
# make -j$(nproc)
# sudo make install

# Remove numpy if needed and reinstall
yes | pip uninstall numpy -y
pip install "numpy<2"
