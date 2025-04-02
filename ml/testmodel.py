import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import threading
import os

yolo = YOLO("yolov8n.pt")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh


frame_width, frame_height = 255, 255
actions = np.array([
# 'Eating_in_classroom' ,
# 'HandRaise' ,
# 'Reading_Book' ,
# 'Sitting_on_Desk' ,
# 'Sleeping',
# 'Writting_on_Textbook'
    "Explaining the Subject",
    "Writing on Board",
    "Using Mobile Phone",
    
])
no_sequences=30
sequence_length = 30

bbox_history = deque(maxlen=10)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,20), thickness=2, circle_radius=2))
    
    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(244,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def smooth_bbox(new_bbox):
    """Smooth bounding box by averaging recent detections."""
    bbox_history.append(new_bbox)
    avg_bbox = np.mean(bbox_history, axis=0)  # Compute the average bounding box
    return tuple(map(int, avg_bbox))

def expand_bbox(x1, y1, x2, y2, scale=1.2):
    """Expand the bounding box by a scale factor."""
    w, h = x2 - x1, y2 - y1
    new_w, new_h = int(w * scale), int(h * scale)
    new_x1 = max(0, x1 - (new_w - w) // 2)
    new_y1 = max(0, y1 - (new_h - h) // 2)
    new_x2 = new_x1 + new_w
    new_y2 = new_y1 + new_h
    return new_x1, new_y1, new_x2, new_y2

label_map = {label:num for num, label in enumerate(actions)}

os.chdir("weights")
model = load_model('teachers.h5')
os.chdir("..")

def resize_and_pad(image, target_size=(255, 255)):
    """Resize image while keeping aspect ratio with padding."""
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_w = (target_size[0] - new_w) // 2
    pad_h = (target_size[1] - new_h) // 2

    padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded


sequence = []
sentence = []
threshold = 0.4

cap = cv2.VideoCapture(0)

frame_lock = threading.Lock()
latest_frame = None
detected_boxes = []
frame_count = 0

# Thread to capture frames continuously
def capture_frames():
    global latest_frame
    while True:
        _, frame = cap.read()
        with frame_lock:
            latest_frame = frame.copy()

capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        frame_count += 1

        # Run YOLO every 5th frame to reduce overhead
        if frame_count % 5 == 0:  
            results = yolo(frame)
            detected_boxes = []

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  
                cls = result.boxes.cls.cpu().numpy()

                for i, box in enumerate(boxes):
                    if int(cls[i]) == 0:  # Only consider 'person' class (ID 0)
                        x1, y1, x2, y2 = map(int, box)

                        # Expand bounding box slightly for stability
                        x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, scale=1.2)
                        detected_boxes.append((x1, y1, x2, y2))

        # Process each detected person
        for x1, y1, x2, y2 in detected_boxes:
            person_crop = frame[y1:y2, x1:x2]
            person_crop_resized = cv2.resize(person_crop, (600, 600))  # Reduce input size for faster processing

            # Pose Estimation
            image, results = mediapipe_detection(person_crop_resized, holistic)
            draw_landmarks(person_crop_resized, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Maintain 30-frame sequence

            # Run model prediction on every frame (instead of waiting for 30 frames)
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                res = np.squeeze(res)  # Ensure it's a 1D array

                # Display each action's probability
                for idx, prob in enumerate(res):
                    text = f"{actions[idx]}: {prob:.2f}"
                    cv2.putText(person_crop_resized, text, (10, 50 + (idx * 30)), 
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

                # Display predicted action
                predicted_action = actions[np.argmax(res)]
                cv2.putText(person_crop_resized, f"Pred: {predicted_action}", (10, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('OpenCV Feed', person_crop_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
