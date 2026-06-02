"""
Generates annotated frame images from the pipeline predictions.
Draws bounding boxes, track IDs, and behaviour labels on frames.
Saves 100 frames spread evenly across the video.

Usage:
    python generate_annotated_frames.py
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
        'output_dir':  'outputs/annotated_frames/cam1',
        'cam_id':      'cam1',
    },
    {
        'video':       'Raw_Videos/CAMERA_2_VIDEO.mov',
        'predictions': 'outputs/predictions_cam2.csv',
        'output_dir':  'outputs/annotated_frames/cam2',
        'cam_id':      'cam2',
    },
]

FRAMES_PER_CAMERA = 50   # 50 per camera = 100 total
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COLOURS = {
    'Focused':      (46,  204, 113),
    'Chatting':     (52,  152, 219),
    'Looking Away': (230, 126,  34),
    'Using Phone':  (231,  76,  60),
    'Unknown':      (149, 165, 166),
}


def draw_frame(frame, frame_predictions, frame_id, cam_id):
    """Draw all person predictions on a single frame."""
    annotated = frame.copy()
    h, w      = frame.shape[:2]

    # Top banner
    cv2.rectangle(annotated, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(annotated,
                f"Frame {frame_id:05d}  |  Camera: {cam_id}  |  "
                f"Persons: {len(frame_predictions)}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (220, 220, 220), 2)

    for _, row in frame_predictions.iterrows():
        label    = row['predicted_label']
        track_id = int(row['track_id'])
        x1       = int(row['bbox_x1'])
        y1       = int(row['bbox_y1'])
        x2       = int(row['bbox_x2'])
        y2       = int(row['bbox_y2'])
        conf     = float(row.get('confidence', 0.0))

        colour = LABEL_COLOURS.get(label, LABEL_COLOURS['Unknown'])

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        # Label background + text
        text = f"ID{track_id}: {label} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 5, th + 10)
        cv2.rectangle(annotated,
                      (x1, label_y - th - 4),
                      (x1 + tw + 4, label_y + 2),
                      colour, -1)
        cv2.putText(annotated, text, (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    # Legend bottom right
    legend_items = list(LABEL_COLOURS.items())[:-1]  # skip Unknown
    lx, ly = w - 230, h - 130
    cv2.rectangle(annotated, (lx - 5, ly - 10),
                  (w - 5, h - 5), (20, 20, 20), -1)
    for i, (lname, lcolour) in enumerate(legend_items):
        y = ly + i * 28
        cv2.rectangle(annotated, (lx, y), (lx + 20, y + 18), lcolour, -1)
        cv2.putText(annotated, lname, (lx + 26, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    return annotated


def process_camera(config):
    video_path   = config['video']
    pred_csv     = config['predictions']
    output_dir   = config['output_dir']
    cam_id       = config['cam_id']

    if not os.path.exists(video_path):
        print(f"  [SKIP] Video not found: {video_path}")
        return 0
    if not os.path.exists(pred_csv):
        print(f"  [SKIP] Predictions not found: {pred_csv}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    # Load predictions
    preds = pd.read_csv(pred_csv)
    preds = preds[preds['predicted_label'] != 'Unknown']

    # Get unique frame IDs that have predictions
    frame_ids = sorted(preds['frame_id'].unique())
    print(f"  {cam_id}: {len(frame_ids)} frames with predictions")

    # Select evenly spaced frames
    if len(frame_ids) <= FRAMES_PER_CAMERA:
        selected_frames = frame_ids
    else:
        indices         = np.linspace(0, len(frame_ids) - 1,
                                      FRAMES_PER_CAMERA, dtype=int)
        selected_frames = [frame_ids[i] for i in indices]

    print(f"  {cam_id}: selecting {len(selected_frames)} frames to annotate...")

    cap     = cv2.VideoCapture(video_path)
    saved   = 0
    target_set = set(selected_frames)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_num not in target_set:
            continue

        frame_preds = preds[preds['frame_id'] == frame_num]
        if len(frame_preds) == 0:
            continue

        annotated = draw_frame(frame, frame_preds, frame_num, cam_id)

        out_path = os.path.join(output_dir, f"frame_{frame_num:05d}.jpg")
        cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

        if saved % 10 == 0:
            print(f"  {cam_id}: saved {saved}/{len(selected_frames)} frames...")

        if saved >= FRAMES_PER_CAMERA:
            break

    cap.release()
    print(f"  ✓ {cam_id}: {saved} annotated frames saved to {output_dir}/")
    return saved


def main():
    print("\n=== Annotated Frame Generator ===\n")

    total = 0
    for config in CAMERAS:
        print(f"Processing {config['cam_id']}...")
        total += process_camera(config)

    print(f"\n{'='*45}")
    print(f"  Total annotated frames saved : {total}")
    print(f"  Camera 1 → outputs/annotated_frames/cam1/")
    print(f"  Camera 2 → outputs/annotated_frames/cam2/")
    print(f"\n  Open these folders and pick your best 2-3 frames")
    print(f"  for your dissertation!")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    main()