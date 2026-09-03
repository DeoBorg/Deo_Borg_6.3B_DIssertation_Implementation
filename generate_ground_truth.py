import cv2
import os
import pandas as pd
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict

# ── CONFIG ───────────────────────────────────────────────────────────────────
ANNOTATIONS_CSV  = "2nd-Dataset/annotations.csv"
FRAMES_DIR       = "2nd-Dataset/frames"
YOLO_MODEL       = "models/best.pt"
OUTPUT_CSV       = "dataset/ground_truth_labels.csv"

CLASS_PERSON = 0
CLASS_LAPTOP = 1
CLASS_PHONE  = 2
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("\n=== Generating Ground Truth Labels with Track IDs ===\n")

    os.makedirs("dataset", exist_ok=True)

    # Load annotations
    ann = pd.read_csv(ANNOTATIONS_CSV)
    print(f"  Loaded {len(ann)} frame annotations")
    print(f"  Cameras: {ann['camera'].unique().tolist()}")
    print(f"  Label distribution:")
    print(ann['label'].value_counts().to_string())

    # Load YOLO
    print(f"\n  Loading YOLOv8: {YOLO_MODEL}")
    yolo = YOLO(YOLO_MODEL)

    results_rows = []

    # Process each camera separately (ByteTrack state is per-camera)
    for camera in ann['camera'].unique():
        print(f"\n  Processing camera: {camera}")
        cam_ann    = ann[ann['camera'] == camera].sort_values('frame_id').reset_index(drop=True)
        cam_frames_dir = os.path.join(FRAMES_DIR, camera)
        tracker    = sv.ByteTrack()

        for _, row in cam_ann.iterrows():
            frame_file = row['frame_file']
            frame_id   = row['frame_id']
            label      = row['label']

            frame_path = os.path.join(cam_frames_dir, frame_file)
            if not os.path.exists(frame_path):
                continue

            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            # Run YOLO detection
            det_raw = yolo(frame, verbose=False)[0]
            dets    = sv.Detections.from_ultralytics(det_raw)
            persons = dets[dets.class_id == CLASS_PERSON]

            # Run ByteTrack
            persons = tracker.update_with_detections(persons)

            if len(persons) == 0:
                continue

            # Save one row per tracked person per frame
            for i, track_id in enumerate(persons.tracker_id):
                results_rows.append({
                    'frame_id':        frame_id,
                    'camera_id':       camera,
                    'track_id':        str(track_id),
                    'behaviour_label': label,
                    'bbox_x1':         int(persons.xyxy[i][0]),
                    'bbox_y1':         int(persons.xyxy[i][1]),
                    'bbox_x2':         int(persons.xyxy[i][2]),
                    'bbox_y2':         int(persons.xyxy[i][3]),
                })

        print(f"    Done — {len([r for r in results_rows if r['camera_id'] == camera])} person-frame pairs")

    # Save
    df = pd.DataFrame(results_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  Saved {len(df)} rows to {OUTPUT_CSV}")
    print(f"\n  Label distribution in ground truth:")
    print(df['behaviour_label'].value_counts().to_string())
    print(f"\n  Unique track IDs per camera:")
    for camera in df['camera_id'].unique():
        n = df[df['camera_id'] == camera]['track_id'].nunique()
        print(f"    {camera}: {n} tracks")

if __name__ == "__main__":
    main()