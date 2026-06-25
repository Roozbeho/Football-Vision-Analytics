#  Football Vision Analytics
### Advanced Computer Vision Pipeline for Real-Time Football Match Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv26-Ultralytics-red?logo=codeforces&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv&logoColor=white)
![Supervision](https://img.shields.io/badge/Supervision-MOT-purple?logo=github&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)
<!-- ![License](https://img.shields.io/badge/License-MIT-yellow.svg) -->

---

## Demo

<p align="center">
  <img src="data/demo.gif" alt="Football Vision Analytics Demo" width="1000">
</p>

<p align="center">
  End-to-end football analytics including player tracking, team assignment, ball possession analysis, and tactical mini-map visualization.
</p>


---

##  Overview
**Football Vision Analytics** is an end-to-end computer vision pipeline designed to extract actionable tactical data from raw football (soccer) broadcast videos. Built on top of **YOLOv8**, **OpenCV**, and the **Supervision** library, this system automatically detects, tracks, and analyzes players, referees, and the ball in real-time. 

It features unsupervised team clustering, perspective transformation (homography) to map 2D camera views to a top-down tactical pitch, and dynamic ball possession tracking.

##  Key Features
-  **Multi-Object Detection & Tracking**: Robust detection of players, goalkeepers, referees, and the ball using custom-trained YOLOv8 models, coupled with the powerful **`supervision`** library for highly efficient Multi-Object Tracking (MOT) and bounding box management across frames.
-  **Automatic Team Assignment**: Unsupervised **K-Means clustering** applied to extracted player jersey colors automatically separates the two teams on the pitch without requiring predefined team kits.
-  **Perspective Transformation (Homography)**: Detects pitch keypoints to calculate a homography matrix, seamlessly mapping broadcast camera coordinates to a 2D top-down tactical mini-map.
-  **Dynamic Ball Possession**: Assigns the ball to the nearest player frame-by-frame to calculate and visualize real-time team possession statistics.
-  **Tactical Mini-Map Overlay**: Renders a synchronized top-down pitch view alongside the broadcast feed, showing exact player positioning, spacing, and movement.
-  **Event Detection**: Integrated spatial logic to detect potential goal-scoring events based on transformed pitch coordinates.
-  **Rich Visualizations**: Overlays bounding boxes, player IDs, team colors, and possession bars directly onto the video feed using `supervision`'s advanced drawing utilities.

---

##  Project Architecture

```text
├── ball_assigner/        # Logic for assigning ball possession to the closest player
├── config.py             # Centralized configuration (paths, thresholds, colors)
├── data/                 # Input videos and datasets
├── main.py               # Main execution pipeline orchestrating all modules
├── requirements.txt      # Python dependencies (ultralytics, opencv-python, supervision, etc.)
├── runs/                 # YOLO weights (object detection & keypoint detection)
├── team_assignment/      # K-Means clustering for player jersey color extraction
├── tests/                # Unit tests for pipeline components
├── tracking/             # Multi-object tracking wrappers utilizing `supervision`
├── train_model.ipynb     # Jupyter notebook for training/fine-tuning YOLO models
├── transformation/       # Camera calibration, homography, and coordinate transformation
├── visualization/        # OpenCV & supervision drawing utilities for frames, pitch maps, and UI
└── yolo.py               # Wrapper for custom YOLO object and keypoint detection models
```


---

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Roozbeho/Football-Vision-Analytics.git
cd Football-Vision-Analytics
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note:** Ensure you have the appropriate CUDA drivers installed if you plan to run inference on a GPU (`device="cuda"`).

---

##  Usage

The pipeline is controlled centrally via the `config.py` file. Before running the script, ensure your paths and settings are correctly configured.

1. Place your input match video inside the `data/` directory.
2. Open `config.py` and update `VIDEO_PATH`, `OUTPUT_PATH`, and ensure the model paths are correct.
3. Run the main execution script:

```bash
python main.py
```

The processed video will be saved to the location specified in `OUTPUT_PATH` using the configured `VIDEO_CODEC` and `OUTPUT_FPS`.

---

##  Training Custom Models

The repository includes `train_model.ipynb`, a comprehensive Jupyter Notebook to fine-tune **YOLOv8** models on custom football datasets. 

The training pipeline utilizes the `ultralytics` API with advanced data augmentations tailored for sports analytics:
- **Mosaic Augmentation** (`1.0`)
- **MixUp** (`0.15`)
- **Copy-Paste** (`0.2`) for instance segmentation/bounding boxes
- **HSV Color Jittering** and **Random Horizontal Flipping**

You can train two separate models:
1. **Object Detector**: For detecting players, referees, goalkeepers, and the ball.
2. **Keypoint/Pose Estimator**: For detecting pitch lines and intersections for homography.

---

##  Configuration

All pipeline behaviors, thresholds, paths, and visualization settings are managed in `config.py`. Below is the default configuration structure:

```python
# --- File Paths ---
OBJECT_MODEL_PATH = "runs/obj_detector/weights/best.pt"
KEYPOINT_MODEL_PATH = "runs/keypoint_detection/weights/best.pt"
VIDEO_PATH = "data/input_video.mp4"
OUTPUT_PATH = "output.mp4"
PITCH_IMG_PATH = "data/top-view-of-green-football-pitch-or-soccer-field-vector.jpg"

# --- Device & Processing ---
DEVICE = "cuda"             # Use "cuda" for GPU or "cpu"
FRAME_SKIP = 1              # Process every Nth frame (1 = no skipping, >1 speeds up processing)
VIDEO_CODEC = "mp4v"        # OpenCV VideoWriter fourcc codec
OUTPUT_FPS = None           # Output FPS (None defaults to matching the input video's FPS)

# --- Detection Thresholds ---
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for object detection
KEYPOINT_CONF_THRESHOLD = 0.5 # Minimum confidence for pitch keypoints

# --- Team & Ball Logic ---
KMEANS_N_CLUSTERS = 2       # Number of teams on the pitch (usually 2)
MAX_BALL_DISTANCE = 70      # Max pixel distance to assign ball possession to a player

# --- Visualization Colors (BGR format) ---
PLAYER_COLORS = [(255, 0, 0), (0, 0, 255)]  # Team 1 (Blue), Team 2 (Red)
GOALKEEPER_COLOR = (255, 126, 255)          # Pinkish
REFEREE_COLOR = (0, 0, 0)                   # Black
BALL_COLOR = (0, 255, 255)                  # Yellow
```

---

##  Contributing
Contributions are welcome! If you have ideas for new features (e.g., passing network generation, speed/acceleration tracking, or action recognition), feel free to open an issue or submit a pull request.
