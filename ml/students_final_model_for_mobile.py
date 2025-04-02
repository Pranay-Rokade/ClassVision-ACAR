import cv2
import os
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
from tensorflow.keras.models import load_model
import time
import torch
from collections import defaultdict


# Load Models
os.chdir('weights')
model = load_model("actionsIncludingSleeping.h5")
yolo_model = YOLO("yolo11n.pt").to("cuda" if torch.cuda.is_available() else "cpu")
os.chdir('..')


# Mediapipe Configuration
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Class Labels
actions = ['Eating in Classroom', 'Hand Raise', 'Reading Book', 'Sitting on Desk', 'Sleeping', 'Writing in Textbook', "Using Phone"]

DISTRACTED_ACTIONS = ['Eating in Classroom', 'Sleeping', 'Using Phone']
PRODUCTIVE_ACTIONS = ['Hand Raise', 'Reading Book', 'Sitting on Desk', 'Writing in Textbook']

label_map = {action: i for i, action in enumerate(actions)}

# dictionaries and Constants
bbox_history = {}
sequence_data = {}
sequence = []
student_sequences = defaultdict(lambda: [])
last_action = {}

SEQUENCE_LENGTH = 30
MODEL_CALL_INTERVAL = 1
FRAME_INTERVAL = 0
THRESHOLD = 0.10




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




def run_multiple_inference(video_path=None):
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    frame_count = 0

    output_video = None
    # if video_path is not None:
    #     base_name = os.path.basename(video_path)
    #     name, ext = os.path.splitext(base_name)
    #     output_path = f"{name}_analyzed{ext}"
         
    #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps = cap.get(cv2.CAP_PROP_FPS)
         
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # if frame_count % FRAME_INTERVAL != 0:
            #     frame_count += 1
            #     continue

            detected_obejets = yolo_model.track(frame, persist=True, classes=[0, 67])  # Detect only humans and phones
            # detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []

            phones = []
            persons = {}
            person_ids_using_phone = []

            for obj in detected_obejets:
                if obj.boxes.id is None:  # Skip if no tracking ID assigned
                    continue
                for box, cls, track_id in zip(obj.boxes.xyxy.cpu().numpy(), obj.boxes.cls.cpu().numpy(), obj.boxes.id):
                    if int(cls) == 67:  # 'cell phone'
                        phones.append(box)
                    elif int(cls) == 0 and track_id is not None:  # 'person' with ID
                        persons[int(track_id)] = box

            if phones and persons:
                for phone in phones:
                    phone_center = get_center(phone)
                    min_distance = float('inf')
                    closest_person_id = None

                    for person_id, person_box in persons.items():
                        person_center = get_center(person_box)
                        distance = abs(phone_center[0] - person_center[0]) + abs(phone_center[1] - person_center[1])

                        if distance < min_distance:
                            min_distance = distance
                            closest_person_id = person_id
                    
                    if closest_person_id is not None:
                        person_ids_using_phone.append(closest_person_id)
                        x1, y1, x2, y2 = map(int, persons[closest_person_id])

                        # Draw bounding box only around phone user
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Using Phone (ID {closest_person_id})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        x1, y1, x2, y2 = map(int, phone)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            
            for track, person in persons.items():
                # if person is identified as using phone then skip
                if track in person_ids_using_phone:
                    continue

                # track id and area extraction of each person
                track_id = track
                x1, y1, x2, y2 = map(int, person)
                h, w, _ = frame.shape
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                # cropped area normalizaion of each person
                x1, y1, x2, y2 = smooth_bbox(track_id, (x1, y1, x2, y2))
                x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, frame, scale=1.2)

                # region of interest of each person
                roi = frame[y1:y2, x1:x2]
                if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                        continue
                person_crop_resized = cv2.resize(roi, (224, 224))  

                # keypoints extraction of each cropped person
                image, results = mediapipe_detection(person_crop_resized, holistic)
                # draw_landmarks(person_crop_resized, results)
                keypoints = extract_keypoints(results)

                # to keep track of each person's keypoints and predictions
                student_sequences[track_id].append(keypoints)
                predictions = {}
                predicted_action = "Sitting on Desk"

                if track_id not in sequence_data:
                    sequence_data[track_id] = []
                sequence_data[track_id].append(keypoints)
                # sequence_data[track_id] = sequence_data[track_id][-SEQUENCE_LENGTH:]

                if len(sequence_data[track_id]) > SEQUENCE_LENGTH:
                    sequence_data[track_id] = sequence_data[track_id][MODEL_CALL_INTERVAL:]  # Remove first MODEO_CALL_INTERVAL frames

                if len(sequence_data[track_id]) == SEQUENCE_LENGTH:
                    res = model.predict(np.expand_dims(sequence_data[track_id], axis=0))
                    res = np.squeeze(res)
                    predicted_action = actions[np.argmax(res)]
                    confidence = res[np.argmax(res)]

                    if confidence > THRESHOLD:  # Confidence threshold
                        last_action[track_id] = predicted_action  # Store last detected action
                    else:
                        last_action[track_id] = "Sitting on Desk"

                if predicted_action in PRODUCTIVE_ACTIONS:
                    # Draw bounding box and show last detected action (avoids flickering)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # action_text = last_action.get(track_id, "Sitting on Desk")
                    cv2.putText(frame, predicted_action, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else :
                    # Draw bounding box and show last detected action (avoids flickering)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # action_text = last_action.get(track_id, "Sitting on Desk")
                    cv2.putText(frame, predicted_action, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # y_offset = 60
                # for action, score in predictions.items():
                #     print(f'{action}: {score:.2f}')

            if output_video is not None:
                 output_video.write(frame)
            else:
                cv2.imshow('Classroom Monitoring', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    if output_video is not None:
         output_video.release()
    cap.release()
    cv2.destroyAllWindows()

run_multiple_inference("https://192.0.0.4:8080/video")
