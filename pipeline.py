"""
Runs the complete smart monitoring system on a video file:
    1. YOLOv8 — detects Person, Laptop, Phone per frame
    2. ByteTrack — assigns consistent track IDs per person
    3. BlazePose — extracts pose features per tracked person
    4. LSTM — classifies behaviour using last 5 frames
    5. Saves predictions to CSV + annotated video

Usage:
    python pipeline.py --video Raw_Videos/CAMERA_1_VIDEO.mp4 --output outputs/predictions_cam1.csv
    python pipeline.py --video Raw_Videos/CAMERA_2_VIDEO.mov --output outputs/predictions_cam2.csv
"""

import cv2
import torch
import numpy as np
import pandas as pd
import mediapipe as mp
import argparse
import os

from ultralytics import YOLO
from collections import defaultdict
import supervision as sv

# ── CONFIG ───────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = "models/best.pt"
LSTM_MODEL_PATH = "models/lstm_best.pt"

SEQUENCE_LENGTH = 5
FEATURE_DIM     = 9
NUM_CLASSES     = 4
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
DROPOUT         = 0.4

PHONE_PROXIMITY_PX  = 200
LAPTOP_PROXIMITY_PX = 350

CLASS_PERSON = 1
CLASS_LAPTOP = 0
CLASS_PHONE  = 2

LABEL_MAP = {
    0: 'Focused',
    1: 'Chatting',
    2: 'Looking Away',
    3: 'Using Phone'
}

LABEL_COLOURS = {
    'Focused':      (46,  204, 113),
    'Chatting':     (52,  152, 219),
    'Looking Away': (230, 126,  34),
    'Using Phone':  (231,  76,  60),
    'Unknown':      (149, 165, 166),
}
# ─────────────────────────────────────────────────────────────────────────────


# ── LSTM Model ───────────────────────────────────────────────────────────────
class BehaviourLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=FEATURE_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0
        )
        self.dropout = torch.nn.Dropout(DROPOUT)
        self.fc      = torch.nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)


# ── Pose Helpers ─────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                       min_detection_confidence=0.4)


def get_landmarks(frame, box):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if not result.pose_landmarks:
        return None
    return result.pose_landmarks.landmark


def estimate_head_pose(lm, crop_shape):
    h, w      = crop_shape[:2]
    nose      = np.array([lm[0].x * w,  lm[0].y * h])
    left_ear  = np.array([lm[7].x * w,  lm[7].y * h])
    right_ear = np.array([lm[8].x * w,  lm[8].y * h])
    left_sh   = np.array([lm[11].x * w, lm[11].y * h])
    right_sh  = np.array([lm[12].x * w, lm[12].y * h])
    ear_mid   = (left_ear + right_ear) / 2
    sh_mid    = (left_sh  + right_sh)  / 2
    yaw       = float(nose[0] - ear_mid[0])
    pitch     = float(sh_mid[1] - nose[1])
    return yaw, pitch


def bbox_centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def pixel_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def peer_facing_score(yaw_a, yaw_b):
    diff = abs(abs(yaw_a - yaw_b) - 180)
    return max(0.0, 1.0 - diff / 180.0)


def build_feature_vector(lm, crop_shape, person_box,
                          phone_boxes, laptop_boxes, other_yaws):
    if lm is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    yaw, pitch = estimate_head_pose(lm, crop_shape)
    person_ctr = bbox_centre(person_box)

    min_phone_dist = float('inf')
    phone_nearby   = 0.0
    for pb in phone_boxes:
        dist = pixel_distance(person_ctr, bbox_centre(pb))
        if dist < min_phone_dist:
            min_phone_dist = dist
    if min_phone_dist < PHONE_PROXIMITY_PX:
        phone_nearby = 1.0
    phone_dist_norm = min(min_phone_dist / 640.0, 1.0) if phone_boxes else 1.0

    laptop_nearby = 0.0
    for lb in laptop_boxes:
        if pixel_distance(person_ctr, bbox_centre(lb)) < LAPTOP_PROXIMITY_PX:
            laptop_nearby = 1.0
            break

    left_wrist_y  = float(lm[15].y)
    right_wrist_y = float(lm[16].y)

    sh_dx          = lm[11].x - lm[12].x
    sh_dy          = lm[11].y - lm[12].y
    shoulder_angle = float(np.degrees(np.arctan2(sh_dy, sh_dx)))

    peer_score = 0.0
    if other_yaws:
        peer_score = max(peer_facing_score(yaw, oy) for oy in other_yaws)

    return np.array([
        yaw / 180.0,
        pitch / 180.0,
        shoulder_angle / 180.0,
        left_wrist_y,
        right_wrist_y,
        phone_nearby,
        phone_dist_norm,
        laptop_nearby,
        peer_score,
    ], dtype=np.float32)


def draw_annotations(frame, person_boxes, track_ids, labels, phone_boxes, laptop_boxes):
    """Draw bounding boxes and behaviour labels on frame."""
    annotated = frame.copy()

    # Draw phone boxes
    for pb in phone_boxes:
        x1, y1, x2, y2 = map(int, pb)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 220), 2)
        cv2.putText(annotated, "Phone", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)

    # Draw laptop boxes
    for lb in laptop_boxes:
        x1, y1, x2, y2 = map(int, lb)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (220, 150, 0), 2)
        cv2.putText(annotated, "Laptop", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 150, 0), 1)

    # Draw person boxes with behaviour labels
    for box, track_id, label in zip(person_boxes, track_ids, labels):
        x1, y1, x2, y2 = map(int, box)
        colour = LABEL_COLOURS.get(label, LABEL_COLOURS['Unknown'])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        text = f"ID{track_id}: {label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return annotated


def run_pipeline(video_path, output_csv, output_video=None, sample_rate=1):
    """
    Run the full inference pipeline on a video.

    Args:
        video_path   : path to input video
        output_csv   : path to save predictions CSV
        output_video : path to save annotated video (None = skip)
        sample_rate  : process every Nth frame (1 = every frame)
    """
    print(f"\n=== Smart Monitoring Pipeline ===")
    print(f"  Video      : {video_path}")
    print(f"  Output CSV : {output_csv}")
    print(f"  Sample rate: every {sample_rate} frame(s)\n")

    # ── Load models ───────────────────────────────────────────────────────────
    print("  Loading models...")
    yolo    = YOLO(YOLO_MODEL_PATH)
    tracker = sv.ByteTrack()

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"  Device: {device}")

    lstm = BehaviourLSTM().to(device)
    lstm.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
    lstm.eval()

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video info : {width}×{height} @ {fps:.1f}fps, {total_frames} frames")

    # ── Optional video writer ─────────────────────────────────────────────────
    writer = None
    if output_video:
        os.makedirs(os.path.dirname(output_video) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        print(f"  Output video: {output_video}")

    # ── Feature buffers per track ─────────────────────────────────────────────
    feature_buffers = defaultdict(list)   # track_id → list of feature vectors
    results_log     = []
    frame_num       = 0
    processed       = 0

    print(f"\n  Processing frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Skip frames based on sample rate
        if frame_num % sample_rate != 0:
            if writer:
                writer.write(frame)
            continue

        processed += 1

        # ── Stage 1: YOLOv8 Detection ─────────────────────────────────────────
        det_raw  = yolo(frame, verbose=False)[0]
        dets     = sv.Detections.from_ultralytics(det_raw)

        persons  = dets[dets.class_id == CLASS_PERSON]
        phones   = dets[dets.class_id == CLASS_PHONE]
        laptops  = dets[dets.class_id == CLASS_LAPTOP]

        phone_boxes  = phones.xyxy.tolist()  if len(phones)  > 0 else []
        laptop_boxes = laptops.xyxy.tolist() if len(laptops) > 0 else []

        # ── Stage 2: ByteTrack ────────────────────────────────────────────────
        persons = tracker.update_with_detections(persons)

        if len(persons) == 0:
            if writer:
                writer.write(frame)
            continue

        # ── Stage 3: BlazePose per person ─────────────────────────────────────
        track_ids   = persons.tracker_id.tolist()
        person_boxes_list = persons.xyxy.tolist()

        all_lms  = []
        all_yaws = []

        for pb in person_boxes_list:
            lm = get_landmarks(frame, pb)
            all_lms.append(lm)
            if lm is not None:
                x1, y1, x2, y2 = map(int, pb)
                cs = frame[max(0,y1):y2, max(0,x1):x2].shape
                if cs[0] > 0 and cs[1] > 0:
                    yaw, _ = estimate_head_pose(lm, cs)
                    all_yaws.append(yaw)
                else:
                    all_yaws.append(None)
            else:
                all_yaws.append(None)

        # ── Stage 4: Feature vectors ──────────────────────────────────────────
        frame_labels = []

        for i, (track_id, pb, lm) in enumerate(
                zip(track_ids, person_boxes_list, all_lms)):

            other_yaws = [y for j, y in enumerate(all_yaws)
                          if j != i and y is not None]

            x1, y1, x2, y2 = map(int, pb)
            crop_shape = frame[max(0,y1):y2, max(0,x1):x2].shape

            fv = build_feature_vector(
                lm, crop_shape, pb,
                phone_boxes, laptop_boxes, other_yaws
            )

            feature_buffers[track_id].append(fv)

            # ── Stage 5: LSTM Classification ──────────────────────────────────
            if len(feature_buffers[track_id]) >= SEQUENCE_LENGTH:
                seq = np.array(
                    feature_buffers[track_id][-SEQUENCE_LENGTH:],
                    dtype=np.float32
                )
                tensor = torch.tensor(seq).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = lstm(tensor)
                    pred   = logits.argmax(dim=1).item()
                    conf   = torch.softmax(logits, dim=1).max().item()
                label = LABEL_MAP[pred]
            else:
                label = 'Unknown'
                conf  = 0.0

            frame_labels.append(label)

            results_log.append({
                'frame_id':        frame_num,
                'track_id':        track_id,
                'predicted_label': label,
                'confidence':      round(conf, 3),
                'bbox_x1':         int(pb[0]),
                'bbox_y1':         int(pb[1]),
                'bbox_x2':         int(pb[2]),
                'bbox_y2':         int(pb[3]),
            })

        # ── Annotate frame ────────────────────────────────────────────────────
        if writer:
            annotated = draw_annotations(
                frame, person_boxes_list, track_ids,
                frame_labels, phone_boxes, laptop_boxes
            )
            writer.write(annotated)

        # Progress
        if processed % 100 == 0:
            pct = frame_num / total_frames * 100
            print(f"  Frame {frame_num}/{total_frames} ({pct:.1f}%) — "
                  f"{len(persons)} persons tracked")

    cap.release()
    if writer:
        writer.release()

    # ── Save predictions ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df = pd.DataFrame(results_log)
    df.to_csv(output_csv, index=False)

    print(f"\n  ✓ Processed {processed} frames")
    print(f"  ✓ Predictions saved to {output_csv}")

    if len(df) > 0:
        print(f"\n  Prediction distribution:")
        dist = df[df['predicted_label'] != 'Unknown']['predicted_label'].value_counts()
        for label, count in dist.items():
            pct = count / len(df) * 100
            print(f"    {label:<15}: {count:>5} ({pct:.1f}%)")

    print(f"\n{'='*40}\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Academic Monitoring — Inference Pipeline"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", default="outputs/predictions.csv",
        help="Path to output predictions CSV"
    )
    parser.add_argument(
        "--annotated_video", default=None,
        help="Path to save annotated video (optional, slow)"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=1,
        help="Process every Nth frame (default=1, use 30 for 1fps from 30fps video)"
    )
    args = parser.parse_args()

    run_pipeline(
        video_path    = args.video,
        output_csv    = args.output,
        output_video  = args.annotated_video,
        sample_rate   = args.sample_rate,
    )