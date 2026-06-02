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

## Results

| Component | Metric | Score |
|---|---|---|
| YOLOv8m (fine-tuned) | mAP@0.5 | 0.948 |
| YOLOv8m (fine-tuned) | F1 | 0.929 |
| LSTM Classifier | Mean Macro F1 (5-fold) | 0.31 ± 0.02 |
| Temporal Stability | Mean Label Switch Rate | 0.090 |

---

## Repository Structure

```
SecondImplementation/
├── models/
│   ├── best.pt               # Fine-tuned YOLOv8m weights
│   └── lstm_best.pt          # Trained LSTM weights
│
├── outputs/
│   ├── evaluation/           # Confusion matrices, charts, classification reports
│   ├── predictions_cam1.csv  # Pipeline predictions — Camera 1
│   ├── predictions_cam2.csv  # Pipeline predictions — Camera 2
│   └── lstm_fold_results.csv # K-Fold cross-validation results
│
├── pipeline.py               # End-to-end inference pipeline
├── train_lstm.py             # LSTM training with SMOTE + K-Fold
├── build_sequences.py        # Feature extraction and sequence building
├── evaluate.py               # Full evaluation and chart generation
├── frame_extractor.py        # Extract frames from video at 1 FPS
├── dataset_splitter.py       # Stratified train/val/test split
├── generate_annotated_frames.py  # Generate annotated frame images
├── find_best_frames.py       # Find most interesting frames for analysis
├── find_clean_frames.py      # Find cleanest frames for dissertation figures
├── manual_behaviour_label.py # Manual behaviour annotation tool
└── regenerate_figure.py      # Regenerate specific dissertation figures
```

---

## Setup & Installation

### Requirements
- Python 3.10+
- macOS (Apple Silicon MPS) or Linux (CUDA GPU recommended for training)

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
            matplotlib seaborn scikit-learn torch torchvision imbalanced-learn
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

> **Note:** Roboflow exports classes alphabetically — `Laptop=0, Person=1, Phone=2`. The scripts use this order.

### Step 3 — Split dataset
```bash
python dataset_splitter.py
# Produces stratified 70/15/15 train/val/test split
```

### Step 4 — Train YOLOv8m (Google Colab recommended)
```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
model.train(data='dataset/data.yaml', epochs=50, imgsz=640, batch=16)
# Download best.pt and place in models/
```

### Step 5 — Build LSTM sequences
```bash
python build_sequences.py
# Requires ground truth behaviour labels in dataset/ground_truth_labels.csv
# Output: dataset/sequences_X.npy, sequences_y.npy
```

### Step 6 — Train LSTM
```bash
python train_lstm.py
# 5-fold cross-validation with SMOTE oversampling
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

### Step 8 — Evaluate
```bash
python evaluate.py
# Output: outputs/evaluation/ — confusion matrices, timelines, charts
```

---

## Dataset

The dataset used in this project is not included in this repository due to privacy concerns (video footage of real participants). It consisted of:

- **11 participants** recorded for ~80 minutes across 2 cameras
- **4,804 frames** extracted at 1 FPS
- **2,397 annotated images** for object detection (Person, Laptop, Phone)
- **1,595 behaviour labels** across 4 classes (manually annotated)

---

## Key Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| Object detector | YOLOv8m | Single-stage, real-time capable, strong mAP |
| Multi-object tracker | ByteTrack | Handles low-confidence detections, robust to occlusion |
| Pose estimator | BlazePose (per-crop) | Receives single-person crops from ByteTrack |
| Behaviour classifier | LSTM | Captures temporal dependencies across frame sequences |
| Class imbalance | SMOTE + weighted loss | Minority classes (Looking Away, Using Phone) underrepresented |
| Validation strategy | 5-Fold CV (LSTM only) | Small behaviour dataset; standard split for YOLOv8 (larger) |

---

## Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 detection |
| `supervision` | ByteTrack tracking + annotation |
| `mediapipe` | BlazePose pose estimation |
| `torch` | LSTM model training |
| `imbalanced-learn` | SMOTE oversampling |
| `scikit-learn` | K-Fold, metrics, stratified splitting |
| `opencv-python` | Frame extraction, annotation rendering |
| `pandas / numpy` | Data processing |
| `matplotlib / seaborn` | Evaluation charts |

---

*BSc (Hons) Multimedia in Software Development — MCAST Institute of Information & Communication Technology*