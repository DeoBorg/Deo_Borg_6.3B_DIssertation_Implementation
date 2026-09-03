"""
annotate_clip.py
----------------
Generates a short annotated video clip showing the full pipeline
(YOLOv8 + ByteTrack + BlazePose + LSTM) running live on a segment
of the source video. Simplified style: bounding boxes + labels only,
no top banner or legend - clean for embedding in a presentation.

Usage:
    python annotate_clip.py
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
import os

from ultralytics import YOLO
from collections import defaultdict
import supervision as sv

# ── CONFIG ───────────────────────────────────────────────────────────────────
VIDEO_PATH       = "Raw_Videos/CAMERA_1_VIDEO.mp4"
OUTPUT_PATH      = "outputs/viva_clip_cam1.mp4"

YOLO_MODEL_PATH  = "models/best.pt"
LSTM_MODEL_PATH  = "models/lstm_best.pt"

# Frame range to extract (10fps video -> 120 frames = 12 seconds)
START_FRAME      = 7560
END_FRAME        = 7680

SEQUENCE_LENGTH  = 5
FEATURE_DIM      = 9
NUM_CLASSES      = 4
HIDDEN_SIZE      = 128
NUM_LAYERS       = 2
DROPOUT          = 0.4

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


def draw_simple_annotations(frame, person_boxes, track_ids, labels,
                             phone_boxes, laptop_boxes):
    """Simplified annotation: boxes + labels only, no banner/legend."""
    annotated = frame.copy()

    # Phone boxes
    for pb in phone_boxes:
        x1, y1, x2, y2 = map(int, pb)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 220), 2)

    # Laptop boxes
    for lb in laptop_boxes:
        x1, y1, x2, y2 = map(int, lb)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (220, 150, 0), 1)

    # Person boxes with behaviour labels
    for box, track_id, label in zip(person_boxes, track_ids, labels):
        x1, y1, x2, y2 = map(int, box)
        colour = LABEL_COLOURS.get(label, LABEL_COLOURS['Unknown'])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 3)

        text = f"ID{track_id}: {label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y = max(y1 - 8, th + 8)
        cv2.rectangle(annotated, (x1, label_y - th - 8),
                      (x1 + tw + 8, label_y + 4), colour, -1)
        cv2.putText(annotated, text, (x1 + 4, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated


def main():
    print("\n=== Generating Viva Annotated Clip ===\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading models...")
    yolo    = YOLO(YOLO_MODEL_PATH)
    tracker = sv.ByteTrack()

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"Device: {device}")

    lstm = BehaviourLSTM().to(device)
    lstm.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
    lstm.eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps}fps")
    print(f"Extracting frames {START_FRAME} to {END_FRAME} "
          f"({(END_FRAME-START_FRAME)/fps:.1f} seconds)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # Pre-roll: feed frames before START_FRAME so LSTM buffer is warmed up
    # (sequence length 5, so feed a few frames before start without writing)
    PREROLL = SEQUENCE_LENGTH + 2

    feature_buffers = defaultdict(list)
    frame_num = 0
    written   = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        if frame_num < START_FRAME - PREROLL:
            continue
        if frame_num > END_FRAME:
            break

        # Stage 1: YOLOv8
        det_raw = yolo(frame, verbose=False)[0]
        dets    = sv.Detections.from_ultralytics(det_raw)

        persons = dets[dets.class_id == CLASS_PERSON]
        phones  = dets[dets.class_id == CLASS_PHONE]
        laptops = dets[dets.class_id == CLASS_LAPTOP]

        phone_boxes  = phones.xyxy.tolist()  if len(phones)  > 0 else []
        laptop_boxes = laptops.xyxy.tolist() if len(laptops) > 0 else []

        # Stage 2: ByteTrack
        persons = tracker.update_with_detections(persons)

        if len(persons) == 0:
            if frame_num >= START_FRAME:
                writer.write(frame)
                written += 1
            continue

        track_ids         = persons.tracker_id.tolist()
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

            if len(feature_buffers[track_id]) >= SEQUENCE_LENGTH:
                seq = np.array(
                    feature_buffers[track_id][-SEQUENCE_LENGTH:],
                    dtype=np.float32
                )
                tensor = torch.tensor(seq).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = lstm(tensor)
                    pred   = logits.argmax(dim=1).item()
                label = LABEL_MAP[pred]
            else:
                label = 'Unknown'

            frame_labels.append(label)

        # Only write frames within the requested range
        if frame_num >= START_FRAME:
            annotated = draw_simple_annotations(
                frame, person_boxes_list, track_ids,
                frame_labels, phone_boxes, laptop_boxes
            )
            writer.write(annotated)
            written += 1

        if frame_num % 30 == 0:
            print(f"  Processed frame {frame_num}/{END_FRAME}")

    cap.release()
    writer.release()

    print(f"\nDone! Wrote {written} frames ({written/fps:.1f} seconds)")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()