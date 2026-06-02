"""
Sequences are built per LABEL SEGMENT per track.
A track with 100 Focused + 12 Using Phone frames
produces BOTH Focused sequences AND Using Phone sequences,
instead of drowning the minority label in majority-vote windows.

Usage:
    python build_sequences.py
"""

import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from ultralytics import YOLO
from collections import defaultdict

# ── CONFIG ───────────────────────────────────────────────────────────────────
FRAMES_CAM1      = "dataset/raw_frames/camera_1"
FRAMES_CAM2      = "dataset/raw_frames/camera_2"
GROUND_TRUTH_CSV = "dataset/ground_truth_labels.csv"
YOLO_MODEL       = "models/best.pt"
OUTPUT_X         = "dataset/sequences_X.npy"
OUTPUT_Y         = "dataset/sequences_y.npy"
OUTPUT_META      = "dataset/sequences_meta.csv"

SEQUENCE_LENGTH  = 5    # reduced from 10 to preserve minority classes
FEATURE_DIM      = 9

PHONE_PROXIMITY_PX  = 200
LAPTOP_PROXIMITY_PX = 350

CLASS_PERSON = 0
CLASS_LAPTOP = 1
CLASS_PHONE  = 2

LABEL_MAP = {
    'Focused':      0,
    'Chatting':     1,
    'Looking Away': 2,
    'Using Phone':  3,
}
# ─────────────────────────────────────────────────────────────────────────────

mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                       min_detection_confidence=0.4)


def bbox_centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def pixel_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


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


def peer_facing_score(yaw_a, yaw_b):
    diff = abs(abs(yaw_a - yaw_b) - 180)
    return max(0.0, 1.0 - diff / 180.0)


def build_feature_vector(lm, crop_shape, person_box,
                          phone_boxes, laptop_boxes, other_yaws):
    if lm is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    yaw, pitch = estimate_head_pose(lm, crop_shape)

    person_ctr     = bbox_centre(person_box)
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


def get_frame_path(frame_id, camera_id):
    cam_dir = FRAMES_CAM1 if camera_id == 'cam1' else FRAMES_CAM2
    path = os.path.join(cam_dir, f"frame_{int(frame_id):05d}.jpg")
    return path if os.path.exists(path) else None


def extract_features_for_frame(frame, person_box, person_idx,
                                phone_boxes, laptop_boxes, all_person_boxes):
    """Extract feature vector for one person in one frame."""
    lm = get_landmarks(frame, person_box)

    other_yaws = []
    for i, pb in enumerate(all_person_boxes):
        if i == person_idx:
            continue
        other_lm = get_landmarks(frame, pb)
        if other_lm is not None:
            x1, y1, x2, y2 = map(int, pb)
            cs = frame[max(0,y1):y2, max(0,x1):x2].shape
            if cs[0] > 0 and cs[1] > 0:
                yaw, _ = estimate_head_pose(other_lm, cs)
                other_yaws.append(yaw)

    x1, y1, x2, y2 = map(int, person_box)
    crop_shape = frame[max(0,y1):y2, max(0,x1):x2].shape

    return build_feature_vector(
        lm, crop_shape, person_box,
        phone_boxes, laptop_boxes, other_yaws
    )


def main():
    print("\n=== Phase 5 v2: Feature Extraction (Per-Label Segments) ===\n")

    gt = pd.read_csv(GROUND_TRUTH_CSV, dtype=str)
    gt = gt[gt['behaviour_label'] != 'ambiguous'].reset_index(drop=True)
    gt['frame_id_int'] = gt['frame_id'].astype(int)
    gt = gt.sort_values(['camera_id', 'track_id', 'frame_id_int'])

    print(f"  Ground truth samples : {len(gt)}")
    print(f"  Label distribution:")
    print(gt['behaviour_label'].value_counts().to_string())

    print(f"\n  Loading YOLOv8 model: {YOLO_MODEL}")
    yolo = YOLO(YOLO_MODEL)

    # ── Step 1: Extract feature vector per annotation row ────────────────────
    print(f"\n  Extracting features per annotation...")
    frame_cache = {}   # cache YOLO detections per frame

    # Store features indexed by (camera_id, track_id, frame_id, label)
    annotation_features = []

    for idx, row in gt.iterrows():
        frame_id  = row['frame_id']
        camera_id = row['camera_id']
        track_id  = row['track_id']
        label     = row['behaviour_label']

        frame_path = get_frame_path(frame_id, camera_id)
        if frame_path is None:
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        cache_key = (frame_id, camera_id)
        if cache_key not in frame_cache:
            results      = yolo(frame, verbose=False)[0]
            person_boxes = []
            phone_boxes  = []
            laptop_boxes = []
            for box in results.boxes:
                cls  = int(box.cls[0])
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                if cls == CLASS_PERSON and conf > 0.35:
                    person_boxes.append(xyxy)
                elif cls == CLASS_PHONE and conf > 0.25:
                    phone_boxes.append(xyxy)
                elif cls == CLASS_LAPTOP and conf > 0.35:
                    laptop_boxes.append(xyxy)
            frame_cache[cache_key] = (person_boxes, phone_boxes, laptop_boxes)

        person_boxes, phone_boxes, laptop_boxes = frame_cache[cache_key]

        if not person_boxes:
            continue

        track_idx  = min(int(track_id) - 1, len(person_boxes) - 1)
        person_box = person_boxes[track_idx]

        fv = extract_features_for_frame(
            frame, person_box, track_idx,
            phone_boxes, laptop_boxes, person_boxes
        )

        annotation_features.append({
            'camera_id': camera_id,
            'track_id':  track_id,
            'frame_id':  int(frame_id),
            'label':     label,
            'feature':   fv
        })

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx+1}/{len(gt)} annotations...")

    print(f"  Features extracted: {len(annotation_features)}")

    # ── Step 2: Build sequences per label segment ────────────────────────────
    # Group by (camera_id, track_id, label) — each group is a label segment
    # Then apply sliding window within each segment
    print(f"\n  Building sequences per label segment (window={SEQUENCE_LENGTH})...")

    X_list    = []
    y_list    = []
    meta_list = []

    # Group annotations
    from itertools import groupby

    # Sort by camera, track, label, frame
    annotation_features.sort(
        key=lambda x: (x['camera_id'], x['track_id'], x['label'], x['frame_id'])
    )

    # Group by (camera_id, track_id, label)
    groups = defaultdict(list)
    for ann in annotation_features:
        key = (ann['camera_id'], ann['track_id'], ann['label'])
        groups[key].append(ann)

    label_seq_counts = defaultdict(int)

    for (camera_id, track_id, label), anns in groups.items():
        if label not in LABEL_MAP:
            continue

        anns.sort(key=lambda x: x['frame_id'])
        features = [a['feature'] for a in anns]
        n        = len(features)

        if n < SEQUENCE_LENGTH:
            # Pad with zeros at the start to still get at least 1 sequence
            pad     = SEQUENCE_LENGTH - n
            features = [np.zeros(FEATURE_DIM, dtype=np.float32)] * pad + features
            n        = SEQUENCE_LENGTH

        # Sliding window within this label segment
        for i in range(n - SEQUENCE_LENGTH + 1):
            window = features[i : i + SEQUENCE_LENGTH]
            X_list.append(np.array(window, dtype=np.float32))
            y_list.append(LABEL_MAP[label])
            meta_list.append({
                'camera_id':   camera_id,
                'track_id':    track_id,
                'label':       label,
                'start_frame': anns[min(i, len(anns)-1)]['frame_id']
            })
            label_seq_counts[label] += 1

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"\n  Sequences built:")
    print(f"    X shape : {X.shape}  (sequences, frames, features)")
    print(f"    y shape : {y.shape}  (sequences,)")
    print(f"\n  Sequence label distribution:")
    label_names_inv = {v: k for k, v in LABEL_MAP.items()}
    for label_id in sorted(set(y.tolist())):
        count = (y == label_id).sum()
        print(f"    {label_names_inv[label_id]:<15}: {count} ({count/len(y)*100:.1f}%)")

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)
    pd.DataFrame(meta_list).to_csv(OUTPUT_META, index=False)

    print(f"\n  Saved:")
    print(f"    {OUTPUT_X}")
    print(f"    {OUTPUT_Y}")
    print(f"    {OUTPUT_META}")
    print(f"\n  Next step: run train_lstm.py")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()