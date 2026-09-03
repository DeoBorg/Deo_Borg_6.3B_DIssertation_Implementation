# Smart Monitoring in Academic Environments
### BSc (Hons) Multimedia in Software Development — MCAST IICT

A multi-camera computer vision system that automatically classifies student behaviour in academic environments using a 4-stage pipeline: object detection, multi-object tracking, pose estimation, and temporal behaviour classification.

---

## Detected Behaviours

| Label | Description |
|---|---|
| **Focused** | Student oriented toward laptop or worksheet |
| **Chatting** | Mutual orientation between two or more students |
| **Looking Away** | Head directed away from task and peers |
| **Using Phone** | Phone detected in proximity with head oriented toward it |

---

## Pipeline Architecture

```
Video Input
    → YOLOv8m        (Person, Laptop, Phone detection)
        → ByteTrack      (Multi-object tracking, consistent IDs)
            → BlazePose      (Pose estimation per tracked person)
                → LSTM           (Behaviour classification over 5-frame sequences)
                    → CSV Output     (frame_id, track_id, predicted_label, confidence)
```

---

## Setup & Installation

### Requirements
- Python 3.10+

### 1. Clone the repository
```bash
git clone https://github.com/DeoBorg/Deo_Borg_6.3B_DIssertation_Implementation.git
cd Deo_Borg_6.3B_DIssertation_Implementation
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install ultralytics supervision mediapipe opencv-python numpy pandas \
            matplotlib seaborn scikit-learn torch torchvision
```

---

## Recreating the Pipeline

### Step 1 — Extract frames from video (1 FPS)
```bash
python frame_extractor.py
# Edit the script to point to your video files
# Output: dataset/raw_frames/camera_1/ and camera_2/
```

### Step 2 — Annotate objects
Upload frames to [Roboflow](https://roboflow.com), annotate `Person`, `Laptop`, `Phone`, export in YOLOv8 format.


### Step 3 — Train YOLOv8m (Google Colab recommended)
```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
model.train(data='dataset/data.yaml', epochs=50, imgsz=640, batch=16)
# Download best.pt and place in models/
```

### Step 4 — Generate ground truth labels
```bash
python generate_ground_truth.py
# Runs YOLOv8 + ByteTrack on extracted frames to assign track IDs
# Output: dataset/ground_truth_labels.csv
```

### Step 5 — Build LSTM sequences
```bash
python build_sequences.py
# Requires dataset/ground_truth_labels.csv
# Output: dataset/sequences_X.npy, sequences_y.npy
```

### Step 6 — Train LSTM
```bash
python train_lstm.py
# 5-fold cross-validation with inverse-frequency class weighting
# Output: models/lstm_best.pt
```

### Step 7 — Run inference pipeline
```bash
python pipeline.py \
  --video Raw_Videos/CAMERA_1_VIDEO.mp4 \
  --output outputs/predictions_cam1.csv \
  --sample_rate 10

python pipeline.py \
  --video Raw_Videos/CAMERA_2_VIDEO.mov \
  --output outputs/predictions_cam2.csv \
  --sample_rate 10
```

---

## Dataset

The datasets used in this project are not included in this repository due to privacy concerns (video footage of real participants). They consisted of:

- **Dataset 1** 
- **11 participants** recorded for ~45 minutes across 2 cameras
- **4,804 frames** extracted at 1 FPS
- **2,397 annotated images** for object detection (Person, Laptop, Phone)

- **Dataset 2** 
- **5 participants** recorded for  ~20 minutes across 2 camera angles
- **2,334 frames** extracted at 1 FPS
- **742 annotated images** for object detection (Person, Laptop, Phone)
- **2,334 behaviour labels** across 4 classes (Focused, Chatting, Looking Away, Using Phone)

---

## Key Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| Object detector | YOLOv8m | Single-stage, real-time capable, strong mAP |
| Multi-object tracker | ByteTrack | Handles low-confidence detections, robust to occlusion |
| Pose estimator | BlazePose (per-crop) | Receives single-person crops from ByteTrack |
| Behaviour classifier | LSTM | Captures temporal dependencies across frame sequences |
| Validation strategy | 5-Fold CV (LSTM only) | Small behaviour dataset; standard 80/10/10 split for YOLOv8 |

---

## Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 detection |
| `supervision` | ByteTrack tracking + annotation |
| `mediapipe` | BlazePose pose estimation |
| `torch` | LSTM model training |
| `scikit-learn` | K-Fold, metrics, stratified splitting |
| `opencv-python` | Frame extraction, annotation rendering |
| `pandas / numpy` | Data processing |
| `matplotlib / seaborn` | Evaluation charts |

---

*BSc (Hons) Multimedia in Software Development — MCAST Institute of Information & Communication Technology*