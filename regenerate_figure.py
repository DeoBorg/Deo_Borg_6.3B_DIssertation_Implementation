import cv2
import pandas as pd
import os

TARGETS = [
    {
        'video':        'Raw_Videos/CAMERA_1_VIDEO.mp4',
        'predictions':  'outputs/predictions_cam1.csv',
        'cam_id':       'cam1',
        'target_frame': 7620,
        'title':        'Study Session - Individual Work (Correct Detections)',
        'output_name':  'figure1_cam1_study_correct.jpg',
    },
    {
        'video':        'Raw_Videos/CAMERA_2_VIDEO.mov',
        'predictions':  'outputs/predictions_cam2.csv',
        'cam_id':       'cam2',
        'target_frame': 6030,
        'title':        'Study Session - Individual Work (Correct Detections)',
        'output_name':  'figure2_cam2_study_correct.jpg',
    },
    {
        'video':        'Raw_Videos/CAMERA_1_VIDEO.mp4',
        'predictions':  'outputs/predictions_cam1.csv',
        'cam_id':       'cam1',
        'target_frame': 22140,
        'title':        'Discussion Session - Mixed Results (Failure Analysis)',
        'output_name':  'figure3_cam1_discussion_failure.jpg',
    },
    {
        'video':        'Raw_Videos/CAMERA_2_VIDEO.mov',
        'predictions':  'outputs/predictions_cam2.csv',
        'cam_id':       'cam2',
        'target_frame': 16560,
        'title':        'Discussion Session - Incorrect Classifications (Failure Analysis)',
        'output_name':  'figure4_cam2_discussion_failure.jpg',
    },
]

OUTPUT_DIR = 'outputs/dissertation_figures'

LABEL_COLOURS = {
    'Focused':      (46,  204, 113),
    'Chatting':     (52,  152, 219),
    'Looking Away': (230, 126,  34),
    'Using Phone':  (231,  76,  60),
    'Unknown':      (149, 165, 166),
}

MIN_CONFIDENCE = 0.40   # lower threshold so all predictions show
MIN_BOX_AREA   = 8000   # slightly lower so we catch all persons
# ─────────────────────────────────────────────────────────────────────────────


def draw_frame(frame, frame_preds, frame_id, cam_id, title):
    annotated = frame.copy()
    h, w      = frame.shape[:2]

    # Top banner
    cv2.rectangle(annotated, (0, 0), (w, 60), (20, 20, 20), -1)
    banner = (f"Smart Academic Monitoring  |  {title}  |  "
              f"Camera: {cam_id}  |  Frame: {frame_id:05d}")
    cv2.putText(annotated, banner, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (220, 220, 220), 2)

    shown = 0
    for _, row in frame_preds.iterrows():
        label    = row['predicted_label']
        track_id = int(row['track_id'])
        x1, y1   = int(row['bbox_x1']), int(row['bbox_y1'])
        x2, y2   = int(row['bbox_x2']), int(row['bbox_y2'])
        conf     = float(row.get('confidence', 0.0))

        if conf < MIN_CONFIDENCE:
            continue
        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            continue

        colour = LABEL_COLOURS.get(label, LABEL_COLOURS['Unknown'])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 3)

        text = f"ID{track_id}: {label} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        label_y = max(y1 - 5, th + 65)
        cv2.rectangle(annotated,
                      (x1, label_y - th - 6),
                      (x1 + tw + 6, label_y + 3),
                      colour, -1)
        cv2.putText(annotated, text, (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        shown += 1

    # Legend bottom right
    legend_items = [
        ('Focused',      LABEL_COLOURS['Focused']),
        ('Chatting',     LABEL_COLOURS['Chatting']),
        ('Looking Away', LABEL_COLOURS['Looking Away']),
        ('Using Phone',  LABEL_COLOURS['Using Phone']),
    ]
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

    return annotated, shown


def process(config):
    video_path   = config['video']
    pred_csv     = config['predictions']
    cam_id       = config['cam_id']
    target_frame = config['target_frame']
    title        = config['title']
    output_name  = config['output_name']

    preds = pd.read_csv(pred_csv)
    preds = preds[preds['predicted_label'] != 'Unknown']
    frame_preds = preds[preds['frame_id'] == target_frame].copy()

    if len(frame_preds) == 0:
        print(f"  [WARN] No predictions for frame {target_frame} in {cam_id}")
        return

    label_dist = frame_preds['predicted_label'].value_counts().to_dict()
    print(f"  Frame {target_frame} ({cam_id}): {label_dist}")

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_num == target_frame:
            annotated, shown = draw_frame(
                frame, frame_preds, target_frame, cam_id, title)
            out_path = os.path.join(OUTPUT_DIR, output_name)
            cv2.imwrite(out_path, annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 97])
            print(f"  Saved ({shown} boxes drawn): {out_path}")
            break
    cap.release()


def main():
    print("\n=== Generating Dissertation Figures ===\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for config in TARGETS:
        print(f"Processing {config['cam_id']} "
              f"frame {config['target_frame']}...")
        process(config)
        print()

    print(f"{'='*50}")
    print(f"  All figures saved to: {OUTPUT_DIR}/")
    print(f"\n  Figure 1 — Correct study session (cam1)")
    print(f"  Figure 2 — Correct study session (cam2)")
    print(f"  Figure 3 — Failure case: mixed results (cam1)")
    print(f"  Figure 4 — Failure case: incorrect chatting (cam2)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()