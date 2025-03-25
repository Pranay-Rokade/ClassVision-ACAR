import cv2
import os
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
from tensorflow.keras.models import load_model
import time

# Load Models
os.chdir('weights')
model = load_model("actionsIncludingSleeping.h5")
os.chdir('..')

yolo = YOLO("yolo11n.pt")

# Mediapipe Configuration
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Class Labels
actions = ['Eating in Classroom', 'Hand Raise', 'Reading Book', 'Sitting on Desk', 'Sleeping', 'Writing in Textbook']
label_map = {action: i for i, action in enumerate(actions)}

# History and Settings
bbox_history = {}
sequence_data = {}
max_sequence_length = 30


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


### ----- VIDEO PROCESSING ----- ###

cap = cv2.VideoCapture(0)
prev_time = time.time()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO Detection
        results = yolo(frame)

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            cls = result.boxes.cls.cpu().numpy()  # Class IDs
            ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else range(len(boxes))  # Track IDs

            for i, box in enumerate(boxes):
                if int(cls[i]) == 0:  # Only process 'person' class (ID 0)
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(ids[i])  # Unique ID for each person

                    # Smooth and expand bounding box
                    x1, y1, x2, y2 = smooth_bbox(track_id, (x1, y1, x2, y2))
                    x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, frame, scale=1.2)

                    # Crop, resize and process student ROI
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop is None or person_crop.size == 0 or person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                        continue
                    person_crop_resized = cv2.resize(person_crop, (224, 224))  # Resize for the action model
                    image, results = mediapipe_detection(person_crop_resized, holistic)
                    draw_landmarks(person_crop_resized, results)

                    # Extract keypoints and append to sequence
                    keypoints = extract_keypoints(results)
                    if track_id not in sequence_data:
                        sequence_data[track_id] = []
                    sequence_data[track_id].append(keypoints)
                    sequence_data[track_id] = sequence_data[track_id][-max_sequence_length:]

                    # Predict action if sequence is long enough
                    if len(sequence_data[track_id]) == max_sequence_length:
                        res = model.predict(np.expand_dims(sequence_data[track_id], axis=0))
                        res = np.squeeze(res)
                        predicted_action = actions[np.argmax(res)]
                        confidence = res[np.argmax(res)]

                        # Draw bounding box and label on main frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{predicted_action} ({confidence:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Draw probability bar for each action
                        for idx, prob in enumerate(res):
                            cv2.putText(frame, f"{actions[idx]}: {prob:.2f}", (x1, y2 + 20 + (idx * 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Show FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display Final Frame
        cv2.imshow('ClassVision - Multiple Student Action Recognition', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
