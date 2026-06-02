"""
Finds the cleanest frames for dissertation figures:
1. A clean "Focused" frame from the study session (early frames)
2. A diverse frame showing multiple behaviours correctly
3. Filters out frames with suspiciously small bounding boxes (legs!)

Usage:
    python find_clean_frames.py
"""

import cv2
import pandas as pd
import numpy as np
import os

CAMERAS = [
    {
        'video':       'Raw_Videos/CAMERA_1_VIDEO.mp4',
        'predictions': 'outputs/predictions_cam1.csv',
        'output_dir':  'outputs/clean_frames/cam1',
        'cam_id':      'cam1',
    },
    {
        'video':       'Raw_Videos/CAMERA_2_VIDEO.mov',
        'predictions': 'outputs/predictions_cam2.csv',
        'output_dir':  'outputs/clean_frames/cam2',
        'cam_id':      'cam2',
    },
]

LABEL_COLOURS = {
    'Focused':      (46,  204, 113),
    'Chatting':     (52,  152, 219),
    'Looking Away': (230, 126,  34),
    'Using Phone':  (231,  76,  60),
    'Unknown':      (149, 165, 166),
}

MIN_BOX_AREA    = 15000   # filter out tiny boxes (legs, partial detections)
MIN_CONFIDENCE  = 0.55    # only show confident predictions
MIN_PERSONS     = 6       # frame must have at least this many persons
# ─────────────────────────────────────────────────────────────────────────────


def box_area(row):
    return (row['bbox_x2'] - row['bbox_x1']) * (row['bbox_y2'] - row['bbox_y1'])


def is_clean_frame(frame_preds):
    """Check if frame has clean, full-body detections."""
    # Filter out tiny boxes
    frame_preds = frame_preds.copy()
    frame_preds['area'] = frame_preds.apply(box_area, axis=1)
    frame_preds = frame_preds[frame_preds['area'] >= MIN_BOX_AREA]

    if len(frame_preds) < MIN_PERSONS:
        return False, frame_preds

    # Check average confidence is high
    avg_conf = frame_preds['confidence'].mean()
    if avg_conf < 0.55:
        return False, frame_preds

    return True, frame_preds


def draw_clean_frame(frame, frame_preds, frame_id, cam_id, title=""):
    annotated = frame.copy()
    h, w      = frame.shape[:2]

    # Top banner
    cv2.rectangle(annotated, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.putText(annotated,
                f"Smart Academic Monitoring  |  {title}  |  "
                f"Camera: {cam_id}  |  Frame: {frame_id:05d}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)

    for _, row in frame_preds.iterrows():
        label    = row['predicted_label']
        track_id = int(row['track_id'])
        x1, y1   = int(row['bbox_x1']), int(row['bbox_y1'])
        x2, y2   = int(row['bbox_x2']), int(row['bbox_y2'])
        conf     = float(row.get('confidence', 0.0))

        if conf < MIN_CONFIDENCE:
            continue

        colour    = LABEL_COLOURS.get(label, LABEL_COLOURS['Unknown'])
        thickness = 3

        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)

        text = f"ID{track_id}: {label} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 5, th + 65)
        cv2.rectangle(annotated,
                      (x1, label_y - th - 6),
                      (x1 + tw + 6, label_y + 3),
                      colour, -1)
        cv2.putText(annotated, text, (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Legend
    legend_items = list(LABEL_COLOURS.items())[:-1]
    lx = w - 225
    ly = h - 145
    cv2.rectangle(annotated, (lx - 8, ly - 18),
                  (w - 8, h - 8), (20, 20, 20), -1)
    cv2.putText(annotated, "Behaviour Key:", (lx, ly + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    for i, (lname, lcolour) in enumerate(legend_items):
        y = ly + 22 + i * 27
        cv2.rectangle(annotated, (lx, y), (lx + 20, y + 18), lcolour, -1)
        cv2.putText(annotated, lname, (lx + 26, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    return annotated


def process_camera(config):
    video_path = config['video']
    pred_csv   = config['predictions']
    output_dir = config['output_dir']
    cam_id     = config['cam_id']

    os.makedirs(output_dir, exist_ok=True)

    preds = pd.read_csv(pred_csv)
    preds = preds[preds['predicted_label'] != 'Unknown']
    preds['area'] = preds.apply(box_area, axis=1)
    preds = preds[preds['area'] >= MIN_BOX_AREA]  # remove tiny boxes

    # ── Category 1: Clean focused frames (early session) ─────────────────────
    early = preds[preds['frame_id'] < 9000]
    focused_frames = []
    for frame_id, group in early.groupby('frame_id'):
        clean, clean_group = is_clean_frame(group)
        if not clean:
            continue
        all_focused = (clean_group['predicted_label'] == 'Focused').all()
        if all_focused and len(clean_group) >= MIN_PERSONS:
            avg_conf = clean_group['confidence'].mean()
            focused_frames.append((frame_id, avg_conf, clean_group))

    focused_frames.sort(key=lambda x: -x[1])
    print(f"  {cam_id}: Found {len(focused_frames)} clean focused frames")

    # ── Category 2: Diverse frames (multiple behaviours, late session) ────────
    late = preds[preds['frame_id'] > 15000]
    diverse_frames = []
    for frame_id, group in late.groupby('frame_id'):
        clean, clean_group = is_clean_frame(group)
        if not clean:
            continue
        unique_labels = set(clean_group['predicted_label'].tolist())
        n_minority    = sum(1 for l in unique_labels
                           if l in ('Using Phone', 'Chatting', 'Looking Away'))
        if n_minority >= 2:
            diverse_frames.append((frame_id, n_minority, clean_group))

    diverse_frames.sort(key=lambda x: -x[1])
    print(f"  {cam_id}: Found {len(diverse_frames)} diverse frames")

    # ── Extract and save frames ───────────────────────────────────────────────
    targets = {}

    # Top 5 clean focused frames
    for frame_id, conf, group in focused_frames[:5]:
        targets[frame_id] = (group, f"Study Session — All Focused")

    # Top 10 diverse frames
    for frame_id, n_min, group in diverse_frames[:10]:
        labels = group['predicted_label'].value_counts().to_dict()
        targets[frame_id] = (group, f"Multiple Behaviours Detected")

    print(f"  {cam_id}: Extracting {len(targets)} frames from video...")

    cap   = cv2.VideoCapture(video_path)
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_num not in targets:
            continue

        group, title    = targets[frame_num]
        annotated       = draw_clean_frame(frame, group, frame_num, cam_id, title)
        labels_str      = '_'.join(sorted(set(group['predicted_label'].tolist())))
        n_persons       = len(group)
        out_path        = os.path.join(
            output_dir,
            f"frame{frame_num:05d}_n{n_persons}_{labels_str}.jpg"
        )
        cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 97])
        saved += 1

        if saved >= len(targets):
            break

    cap.release()
    print(f"  ✓ {cam_id}: saved {saved} clean frames to {output_dir}/\n")


def main():
    print("\n=== Clean Frame Finder ===\n")
    print(f"  Filtering: min box area={MIN_BOX_AREA}px², "
          f"min confidence={MIN_CONFIDENCE}, "
          f"min persons={MIN_PERSONS}\n")

    for config in CAMERAS:
        process_camera(config)

    print(f"{'='*50}")
    print(f"  Frames saved to:")
    print(f"    outputs/clean_frames/cam1/")
    print(f"    outputs/clean_frames/cam2/")
    print(f"\n  Filename format: frame_N-persons_labels.jpg")
    print(f"  For dissertation pick:")
    print(f"    1. A frame named *_Focused.jpg (clean study session)")
    print(f"    2. A frame with multiple labels in filename")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()