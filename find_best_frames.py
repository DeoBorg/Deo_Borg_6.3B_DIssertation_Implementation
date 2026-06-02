"""
Hunts through predictions to find the best frames where
minority classes (Chatting, Using Phone, Looking Away) were detected.
Then generates high quality annotated images of those frames.

Usage:
    python find_best_frames.py
"""

import cv2
import pandas as pd
import numpy as np
import os

# ── CONFIG ───────────────────────────────────────────────────────────────────
CAMERAS = [
    {
        'video':       'Raw_Videos/CAMERA_1_VIDEO.mp4',
        'predictions': 'outputs/predictions_cam1.csv',
        'output_dir':  'outputs/best_frames/cam1',
        'cam_id':      'cam1',
    },
    {
        'video':       'Raw_Videos/CAMERA_2_VIDEO.mov',
        'predictions': 'outputs/predictions_cam2.csv',
        'output_dir':  'outputs/best_frames/cam2',
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

MIN_CONFIDENCE = 0.40   # only show predictions above this confidence
# ─────────────────────────────────────────────────────────────────────────────


def score_frame(frame_preds):
    """
    Score a frame by how interesting it is for the dissertation.
    Higher score = more diverse behaviours + higher confidence.
    """
    labels    = frame_preds['predicted_label'].tolist()
    confs     = frame_preds['confidence'].tolist()
    unique    = set(labels)

    # Bonus for having minority classes
    score = 0
    if 'Using Phone'    in unique: score += 10
    if 'Chatting'       in unique: score += 8
    if 'Looking Away'   in unique: score += 6
    if len(unique)      >= 3:      score += 5   # multiple behaviours in one frame
    if len(unique)      >= 2:      score += 3

    # Bonus for high confidence on minority classes
    for label, conf in zip(labels, confs):
        if label in ('Using Phone', 'Chatting', 'Looking Away'):
            score += conf * 5

    # Bonus for more people detected
    score += len(frame_preds) * 0.5

    return score


def draw_annotated_frame(frame, frame_preds, frame_id, cam_id):
    """Draw rich annotations on a frame."""
    annotated = frame.copy()
    h, w      = frame.shape[:2]

    # Top banner
    cv2.rectangle(annotated, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.putText(annotated,
                f"Smart Academic Monitoring  |  Frame {frame_id:05d}  "
                f"|  Camera: {cam_id}  |  Persons detected: {len(frame_preds)}",
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    for _, row in frame_preds.iterrows():
        label    = row['predicted_label']
        track_id = int(row['track_id'])
        x1       = int(row['bbox_x1'])
        y1       = int(row['bbox_y1'])
        x2       = int(row['bbox_x2'])
        y2       = int(row['bbox_y2'])
        conf     = float(row.get('confidence', 0.0))

        if conf < MIN_CONFIDENCE and label != 'Focused':
            continue

        colour    = LABEL_COLOURS.get(label, LABEL_COLOURS['Unknown'])
        thickness = 3 if label != 'Focused' else 2

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)

        # Label tag
        text = f"ID{track_id}: {label} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1 - 5, th + 60)
        cv2.rectangle(annotated,
                      (x1, label_y - th - 5),
                      (x1 + tw + 6, label_y + 3),
                      colour, -1)
        cv2.putText(annotated, text, (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)

    # Legend bottom right
    legend_items = [
        ('Focused',      LABEL_COLOURS['Focused']),
        ('Chatting',     LABEL_COLOURS['Chatting']),
        ('Looking Away', LABEL_COLOURS['Looking Away']),
        ('Using Phone',  LABEL_COLOURS['Using Phone']),
    ]
    lx = w - 220
    ly = h - 140
    cv2.rectangle(annotated, (lx - 8, ly - 15),
                  (w - 8, h - 8), (20, 20, 20), -1)
    cv2.putText(annotated, "Behaviour Key:", (lx, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    for i, (lname, lcolour) in enumerate(legend_items):
        y = ly + 25 + i * 26
        cv2.rectangle(annotated, (lx, y), (lx + 18, y + 16), lcolour, -1)
        cv2.putText(annotated, lname, (lx + 24, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    return annotated


def find_and_save_best_frames(config, n_frames=25):
    video_path = config['video']
    pred_csv   = config['predictions']
    output_dir = config['output_dir']
    cam_id     = config['cam_id']

    if not os.path.exists(video_path) or not os.path.exists(pred_csv):
        print(f"  [SKIP] Missing files for {cam_id}")
        return

    os.makedirs(output_dir, exist_ok=True)

    preds = pd.read_csv(pred_csv)
    preds = preds[preds['predicted_label'] != 'Unknown']

    # Score every frame and sort by score
    frame_scores = []
    for frame_id, group in preds.groupby('frame_id'):
        score = score_frame(group)
        frame_scores.append((frame_id, score, group))

    frame_scores.sort(key=lambda x: -x[1])   # highest score first

    print(f"\n  {cam_id} — Top 10 frame scores:")
    for fid, score, group in frame_scores[:10]:
        labels = group['predicted_label'].value_counts().to_dict()
        print(f"    Frame {fid:05d}: score={score:.1f}  labels={labels}")

    # Select top N frames
    top_frames = frame_scores[:n_frames]
    target_ids = {fid: (score, group) for fid, score, group in top_frames}

    print(f"\n  Extracting and annotating {len(target_ids)} frames from video...")

    cap   = cv2.VideoCapture(video_path)
    saved = 0

    while cap.isOpened() and saved < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_num not in target_ids:
            continue

        score, frame_preds = target_ids[frame_num]
        annotated = draw_annotated_frame(frame, frame_preds, frame_num, cam_id)

        labels_in_frame = frame_preds['predicted_label'].value_counts().to_dict()
        label_str = '_'.join(sorted(labels_in_frame.keys()))
        out_path  = os.path.join(
            output_dir,
            f"score{score:.0f}_frame{frame_num:05d}_{label_str}.jpg"
        )
        cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 97])
        saved += 1

    cap.release()
    print(f"  ✓ Saved {saved} best frames to {output_dir}/")


def main():
    print("\n=== Best Frame Finder ===\n")
    print("Scoring frames by behaviour diversity and confidence...\n")

    for config in CAMERAS:
        print(f"Processing {config['cam_id']}...")
        find_and_save_best_frames(config, n_frames=25)

    print(f"\n{'='*50}")
    print(f"  Best frames saved to:")
    print(f"    outputs/best_frames/cam1/")
    print(f"    outputs/best_frames/cam2/")
    print(f"\n  Files are named: score_frame_labels.jpg")
    print(f"  Higher score = more interesting frame for dissertation")
    print(f"  Pick frames that show multiple colours (behaviours)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()