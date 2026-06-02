"""
manual_label_tool.py
--------------------
Manually label 150 stratified frames for ground truth behaviour annotation.

Frame selection:
  - 50 frames from study session   (frames 0–900,    camera 1)
  - 50 frames from discussion      (frames 901–2403, camera 1)
  - 50 frames mixed                (random across both cameras)

For each frame you will label EVERY person visible one by one.
The person being labelled is highlighted with a bright box.
Everyone else is dimmed.

Controls:
    F = Focused
    C = Chatting
    L = Looking Away
    P = Using Phone
    S = Skip this person (ambiguous)
    Q = Quit and save progress

Progress saves after every person — quit and resume anytime.

Usage:
    python manual_label_tool.py
"""

import cv2
import csv
import os
import random
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# ── CONFIG ───────────────────────────────────────────────────────────────────
FRAMES_CAM1     = "dataset/raw_frames/camera_1"
FRAMES_CAM2     = "dataset/raw_frames/camera_2"
OUTPUT_CSV      = "dataset/ground_truth_labels.csv"
PROGRESS_CSV    = "dataset/manual_label_progress.csv"
YOLO_MODEL      = "yolov8n.pt"

STUDY_END_FRAME = 900    # approximate end of individual study session
N_STUDY         = 50     # frames to sample from study session
N_DISCUSS       = 50     # frames to sample from discussion session
N_MIXED         = 50     # frames to sample mixed across both cameras

WINDOW_W = 1100
WINDOW_H = 700
# ─────────────────────────────────────────────────────────────────────────────

LABEL_KEYS = {
    ord('f'): 'Focused',       ord('F'): 'Focused',
    ord('c'): 'Chatting',      ord('C'): 'Chatting',
    ord('l'): 'Looking Away',  ord('L'): 'Looking Away',
    ord('p'): 'Using Phone',   ord('P'): 'Using Phone',
    ord('s'): 'ambiguous',     ord('S'): 'ambiguous',
    ord('q'): 'QUIT',          ord('Q'): 'QUIT',
}

LABEL_COLOURS = {
    'Focused':      (46,  204, 113),
    'Chatting':     (52,  152, 219),
    'Looking Away': (230, 126,  34),
    'Using Phone':  (231,  76,  60),
    'ambiguous':    (149, 165, 166),
}

CLASS_PERSON = 0
CLASS_PHONE  = 67
CLASS_LAPTOP = 63


def get_frame_number(filename):
    return int(os.path.splitext(filename)[0].replace("frame_", ""))


def select_frames():
    """Select 150 stratified frames across study/discussion/mixed."""
    random.seed(42)

    cam1_files = sorted([f for f in os.listdir(FRAMES_CAM1)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    cam2_files = sorted([f for f in os.listdir(FRAMES_CAM2)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # Study session — cam1 frames 0 to STUDY_END_FRAME
    study_pool = [f for f in cam1_files
                  if get_frame_number(f) <= STUDY_END_FRAME]

    # Discussion session — cam1 frames after STUDY_END_FRAME
    discuss_pool = [f for f in cam1_files
                    if get_frame_number(f) > STUDY_END_FRAME]

    # Mixed — random from both cameras
    mixed_pool = (
        [(f, FRAMES_CAM1) for f in cam1_files] +
        [(f, FRAMES_CAM2) for f in cam2_files]
    )

    study_sample   = random.sample(study_pool,   min(N_STUDY,   len(study_pool)))
    discuss_sample = random.sample(discuss_pool,  min(N_DISCUSS, len(discuss_pool)))
    mixed_sample   = random.sample(mixed_pool,    min(N_MIXED,   len(mixed_pool)))

    # Build unified list: (frame_path, segment_label)
    selected = []
    for f in study_sample:
        selected.append((os.path.join(FRAMES_CAM1, f), "study", "cam1"))
    for f in discuss_sample:
        selected.append((os.path.join(FRAMES_CAM1, f), "discussion", "cam1"))
    for f, cam_dir in mixed_sample:
        cam_id = "cam1" if cam_dir == FRAMES_CAM1 else "cam2"
        selected.append((os.path.join(cam_dir, f), "mixed", cam_id))

    # Shuffle so you don't do all study frames in a row
    random.shuffle(selected)
    return selected


def detect_persons(frame, yolo_model):
    """Run YOLOv8 and return person, phone, laptop boxes."""
    results      = yolo_model(frame, verbose=False)[0]
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
    return person_boxes, phone_boxes, laptop_boxes


def draw_frame(frame, all_boxes, phone_boxes, laptop_boxes,
               highlight_box, person_idx, total_persons,
               frame_num, camera_id, segment,
               labelled_count, total_to_label):
    """Draw dimmed frame with highlighted current person."""
    h, w   = frame.shape[:2]
    result = (frame.astype(np.float32) * 0.35).astype(np.uint8)

    # Restore brightness for highlighted person
    if highlight_box:
        x1, y1, x2, y2 = [int(v) for v in highlight_box]
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(w, x2); y2c = min(h, y2)
        result[y1c:y2c, x1c:x2c] = frame[y1c:y2c, x1c:x2c]

    # Draw phone boxes in red
    for pb in phone_boxes:
        bx1, by1, bx2, by2 = map(int, pb)
        cv2.rectangle(result, (bx1, by1), (bx2, by2), (0, 0, 220), 2)
        cv2.putText(result, "PHONE", (bx1, by1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)

    # Draw laptop boxes in blue
    for lb in laptop_boxes:
        bx1, by1, bx2, by2 = map(int, lb)
        cv2.rectangle(result, (bx1, by1), (bx2, by2), (220, 150, 0), 2)
        cv2.putText(result, "LAPTOP", (bx1, by1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 150, 0), 1)

    # Grey boxes for other persons
    for box in all_boxes:
        if box != highlight_box:
            bx1, by1, bx2, by2 = map(int, box)
            cv2.rectangle(result, (bx1, by1), (bx2, by2), (100, 100, 100), 1)

    # Bright yellow box for current person
    if highlight_box:
        x1, y1, x2, y2 = map(int, highlight_box)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 230, 255), 3)
        label_text = f"Person {person_idx}/{total_persons} — label this one"
        (tw, th), _ = cv2.getTextSize(label_text,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lx = max(0, x1)
        ly = max(th + 6, y1 - 5)
        cv2.rectangle(result, (lx, ly - th - 4),
                      (lx + tw + 6, ly + 2), (0, 230, 255), -1)
        cv2.putText(result, label_text, (lx + 3, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

    # Top HUD
    cv2.rectangle(result, (0, 0), (w, 110), (15, 15, 15), -1)
    cv2.putText(result,
                f"Frame: {frame_num:05d}  |  Camera: {camera_id}  |  Segment: {segment}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)
    cv2.putText(result,
                f"Person {person_idx} of {total_persons} in this frame",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
    cv2.putText(result,
                f"Overall progress: {labelled_count} persons labelled across all frames",
                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 170, 170), 1)

    # Bottom key guide
    cv2.rectangle(result, (0, h - 44), (w, h), (15, 15, 15), -1)
    cv2.putText(result,
                "  F=Focused   C=Chatting   L=LookingAway   P=UsingPhone   S=Skip   Q=Quit&Save",
                (5, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

    return result


def load_progress():
    """Load previously labelled persons."""
    if os.path.exists(PROGRESS_CSV):
        done = set()
        rows = []
        with open(PROGRESS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                done.add((row['frame_path'], row['track_id']))
        print(f"  Resuming — {len(rows)} persons already labelled")
        return rows, done
    return [], set()


def save_progress(rows):
    if not rows:
        return
    fieldnames = ['frame_path', 'frame_id', 'camera_id', 'segment',
                  'track_id', 'behaviour_label']
    with open(PROGRESS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finalise(rows):
    """Write clean ground truth CSV."""
    fieldnames = ['frame_id', 'camera_id', 'segment',
                  'track_id', 'behaviour_label', 'auto_labelled']
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row['behaviour_label'] == 'ambiguous':
                continue   # exclude ambiguous from ground truth
            writer.writerow({
                'frame_id':        row['frame_id'],
                'camera_id':       row['camera_id'],
                'segment':         row['segment'],
                'track_id':        row['track_id'],
                'behaviour_label': row['behaviour_label'],
                'auto_labelled':   0
            })
    print(f"\n  ✓ Ground truth saved to {OUTPUT_CSV}")


def main():
    print("\n=== Manual Ground Truth Labeller ===\n")
    print("  Selecting 150 stratified frames...")

    selected_frames = select_frames()
    print(f"  Selected {len(selected_frames)} frames")
    print(f"  Loading YOLOv8 for person detection...\n")

    yolo_model = YOLO(YOLO_MODEL)

    progress_rows, already_done = load_progress()

    cv2.namedWindow("Manual Label Tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Manual Label Tool", WINDOW_W, WINDOW_H)

    labelled_count = len(already_done)
    frames_done    = 0

    for frame_path, segment, camera_id in selected_frames:
        if not os.path.exists(frame_path):
            continue

        frame    = cv2.imread(frame_path)
        if frame is None:
            continue

        frame_num    = get_frame_number(os.path.basename(frame_path))
        person_boxes, phone_boxes, laptop_boxes = detect_persons(frame, yolo_model)

        if not person_boxes:
            continue

        frames_done += 1
        print(f"  Frame {frame_num:05d} ({segment}) — {len(person_boxes)} people detected")

        # Label each person in this frame
        for p_idx, pb in enumerate(person_boxes, start=1):
            track_id = str(p_idx)
            key      = (frame_path, track_id)

            if key in already_done:
                continue   # already labelled in a previous session

            display = draw_frame(
                frame, person_boxes, phone_boxes, laptop_boxes,
                pb, p_idx, len(person_boxes),
                frame_num, camera_id, segment,
                labelled_count, labelled_count
            )
            cv2.imshow("Manual Label Tool", display)

            while True:
                k = cv2.waitKey(0) & 0xFF
                if k in LABEL_KEYS:
                    action = LABEL_KEYS[k]

                    if action == 'QUIT':
                        save_progress(progress_rows)
                        finalise(progress_rows)
                        print(f"\n  Saved. {labelled_count} persons labelled.")
                        print(f"  Run again to continue from where you left off.")
                        cv2.destroyAllWindows()
                        return

                    progress_rows.append({
                        'frame_path':     frame_path,
                        'frame_id':       f"{frame_num:05d}",
                        'camera_id':      camera_id,
                        'segment':        segment,
                        'track_id':       track_id,
                        'behaviour_label': action
                    })
                    already_done.add(key)
                    labelled_count += 1

                    # Auto-save every 50 labels
                    if labelled_count % 50 == 0:
                        save_progress(progress_rows)
                        print(f"  Auto-saved at {labelled_count} labels")
                    break
                else:
                    print("  Invalid key — use F, C, L, P, S, or Q")

    cv2.destroyAllWindows()
    save_progress(progress_rows)
    finalise(progress_rows)

    # Summary
    label_counts = {}
    for row in progress_rows:
        l = row['behaviour_label']
        label_counts[l] = label_counts.get(l, 0) + 1

    print(f"\n{'='*50}")
    print(f"  Labelling complete!")
    print(f"  Total persons labelled : {labelled_count}")
    print(f"\n  Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {label:<15} : {count}")
    print(f"\n  Ground truth saved to : {OUTPUT_CSV}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()