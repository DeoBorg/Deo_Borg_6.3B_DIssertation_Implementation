"""
frame_extractor.py
------------------
Extracts frames at 1 FPS from two camera video files.
Output structure:
    dataset/raw_frames/camera_1/frame_00001.jpg
    dataset/raw_frames/camera_2/frame_00001.jpg

Usage:
    python frame_extractor.py
"""

import cv2
import os

# ── CONFIG ─────────────────────────────────────────────────────────────────
# Update these two filenames to match your actual video file names
CAMERA_1_VIDEO = "Raw_Videos/CAMERA_1_VIDEO.mp4"
CAMERA_2_VIDEO = "Raw_Videos/CAMERA_2_VIDEO.mov"

OUTPUT_BASE_DIR = "dataset/raw_frames"
FPS_TARGET = 1   # extract 1 frame per second
# ───────────────────────────────────────────────────────────────────────────


def extract_frames(video_path, output_dir, fps_target=1, camera_label=""):
    """
    Extract frames from a video file at the target FPS rate.

    Args:
        video_path  : path to the video file
        output_dir  : folder to save extracted frames
        fps_target  : how many frames per second to extract (default 1)
        camera_label: label used in print messages (e.g. "Camera 1")
    """

    # Check the video file exists before doing anything
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        print(f"        Make sure the file is in your SecondImplementation folder")
        print(f"        and that the filename in CONFIG matches exactly (including extension).")
        return 0

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0

    # Get source video properties
    source_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_secs = int(total_frames / source_fps) if source_fps > 0 else 0
    interval      = max(1, int(round(source_fps / fps_target)))

    print(f"\n{'─'*50}")
    print(f"  {camera_label}")
    print(f"  File       : {video_path}")
    print(f"  Source FPS : {source_fps:.2f}")
    print(f"  Duration   : {duration_secs // 60}m {duration_secs % 60}s")
    print(f"  Total frames in video : {total_frames}")
    print(f"  Extracting every {interval} frames (≈ {fps_target} FPS)")
    print(f"  Expected output frames: ~{duration_secs * fps_target}")
    print(f"  Output dir : {output_dir}")
    print(f"{'─'*50}")

    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    saved       = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1

            # Print progress every 100 saved frames
            if saved % 100 == 0:
                print(f"  ... saved {saved} frames so far")

        frame_count += 1

    cap.release()
    print(f"\n  ✓ Done — extracted {saved} frames from {video_path}\n")
    return saved


def main():
    print("\n=== Frame Extractor ===")
    print(f"Target: {FPS_TARGET} frame(s) per second\n")

    cam1_out = os.path.join(OUTPUT_BASE_DIR, "camera_1")
    cam2_out = os.path.join(OUTPUT_BASE_DIR, "camera_2")

    saved_1 = extract_frames(CAMERA_1_VIDEO, cam1_out, FPS_TARGET, "Camera 1")
    saved_2 = extract_frames(CAMERA_2_VIDEO, cam2_out, FPS_TARGET, "Camera 2")

    print("=" * 50)
    print(f"  Camera 1 frames saved : {saved_1}")
    print(f"  Camera 2 frames saved : {saved_2}")
    print(f"  Total frames          : {saved_1 + saved_2}")
    print(f"\n  Frames saved to: {os.path.abspath(OUTPUT_BASE_DIR)}")
    print("=" * 50)


if __name__ == "__main__":
    main()