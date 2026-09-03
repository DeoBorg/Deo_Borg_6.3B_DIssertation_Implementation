# debug_features.py
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import mediapipe as mp

YOLO_MODEL_PATH = "models/best.pt"
CLASS_PERSON = 0
CLASS_LAPTOP = 1
CLASS_PHONE  = 2
PHONE_PROXIMITY_PX = 200

yolo    = YOLO(YOLO_MODEL_PATH)
tracker = sv.ByteTrack()
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(static_image_mode=False, model_complexity=1)

cap = cv2.VideoCapture("2nd-Dataset/Angle1.mov")
cap.set(cv2.CAP_PROP_POS_MSEC, 313 * 1000)

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break

    det_raw = yolo(frame, verbose=False)[0]
    dets    = sv.Detections.from_ultralytics(det_raw)
    persons = dets[dets.class_id == CLASS_PERSON]
    phones  = dets[dets.class_id == CLASS_PHONE]
    persons = tracker.update_with_detections(persons)

    phone_boxes = phones.xyxy.tolist() if len(phones) > 0 else []
    print(f"Frame {i+1}: {len(persons)} persons, {len(phone_boxes)} phones detected")

    for j, pb in enumerate(persons.xyxy.tolist()):
        pctr = ((pb[0]+pb[2])/2, (pb[1]+pb[3])/2)
        for ph in phone_boxes:
            phctr = ((ph[0]+ph[2])/2, (ph[1]+ph[3])/2)
            dist  = np.sqrt((pctr[0]-phctr[0])**2 + (pctr[1]-phctr[1])**2)
            print(f"  Person {j+1} → Phone distance: {dist:.1f}px (threshold: {PHONE_PROXIMITY_PX}px)")

cap.release()