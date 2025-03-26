import cv2
import os
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
from tensorflow.keras.models import load_model
import time
import torch

# Load Models
os.chdir('weights')
model = load_model("actionsIncludingSleeping.h5")
yolo = YOLO("yolo11n.pt").to("cuda" if torch.cuda.is_available() else "cpu")
os.chdir('..')


# Mediapipe Configuration
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Class Labels
actions = ['Eating in Classroom', 'Hand Raise', 'Reading Book', 'Sitting on Desk', 'Sleeping', 'Writing in Textbook', "Using Phone"]
label_map = {action: i for i, action in enumerate(actions)}

# History and Constants
bbox_history = {}
sequence_data = {}
SEQUENCE_LENGTH = 30
MODEL_CALL_INTERVAL = 1
FRAME_INTERVAL = 0


### ----- HELPER FUNCTIONS ----- ###

def mediapipe_detection(image, model):
    """Detect landmarks using Mediapipe."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    """Draw pose, face, and hand landmarks."""
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    """Extract keypoints for pose, face, and hands."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
                    ).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
                    ).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
                  ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
                  ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def smooth_bbox(id, new_bbox):
    """Smooth bounding box for each detected student."""
    if id not in bbox_history:
        bbox_history[id] = deque(maxlen=10)
    bbox_history[id].append(new_bbox)
    if len(bbox_history[id]) == 0:
        return new_bbox
    avg_bbox = np.round(np.mean(bbox_history[id], axis=0)).astype(int)
    return avg_bbox

def expand_bbox(x1, y1, x2, y2, frame, scale=1.2):
    """Expand bounding box by a scale factor and ensure boundaries."""
    h, w, _ = frame.shape
    w_bbox, h_bbox = x2 - x1, y2 - y1
    new_w, new_h = int(w_bbox * scale), int(h_bbox * scale)
    new_x1 = max(0, x1 - (new_w - w_bbox) // 2)
    new_y1 = max(0, y1 - (new_h - h_bbox) // 2)
    new_x2 = min(w, new_x1 + new_w)
    new_y2 = min(h, new_y1 + new_h)
    return new_x1, new_y1, new_x2, new_y2

def get_center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


### ----- VIDEO PROCESSING ----- ###

