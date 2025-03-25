import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import os
from collections import defaultdict

actions = np.array(['Eating_in_classroom', 'HandRaise', 'Reading_Book','Sitting_on_Desk','Writting_on_Textbook'])
yolo_model = YOLO("yolo11n.pt")
os.chdir('weights')
models = {action: load_model(f'{action}.h5') for action in actions}
os.chdir('..')

no_sequences=50
sequence_length = 30

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh

MODEL_CALL_INTERVAL = 1
FRAME_INTERVAL = 0
SEQUENCE_LENGTH = 30
FRAME_SIZE = (255, 255)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face.FACEMESH_TESSELATION)
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

label_map = {label:num for num, label in enumerate(actions)}

sequence = []
threshold = 0.10
student_sequences = defaultdict(lambda: [])


# Store the last detected action for each student
last_action = {}

def run_multiple_inference(video_path=None):
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    frame_count = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # if frame_count % FRAME_INTERVAL != 0:
            #     frame_count += 1
            #     continue

            results = yolo_model.track(frame, persist=True, classes=[0])  # Detect only humans
            detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []

            for box in detections:
                x1, y1, x2, y2, track_id, _ = map(int, box[:6])

                h, w, _ = frame.shape
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                roi = frame[y1:y2, x1:x2]

                image, results = mediapipe_detection(roi, holistic)
                keypoints = extract_keypoints(results)

                student_sequences[track_id].append(keypoints)
                predictions = {}
                if len(student_sequences[track_id]) > SEQUENCE_LENGTH:
                    student_sequences[track_id] = student_sequences[track_id][MODEL_CALL_INTERVAL:]  # Remove first 15 frames

                if len(student_sequences[track_id]) == SEQUENCE_LENGTH:
                    predictions = {action: model.predict(np.expand_dims(student_sequences[track_id], axis=0))[0][0] for action, model in models.items()}
                    best_action = max(predictions, key=predictions.get)
                    best_score = predictions[best_action]

                    if best_score > threshold:  # Confidence threshold
                        last_action[track_id] = best_action  # Store last detected action
                    else:
                        last_action[track_id] = "Sitting on Desk"

                # Draw bounding box and show last detected action (avoids flickering)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                action_text = last_action.get(track_id, "Sitting on Desk")
                cv2.putText(frame, action_text, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                y_offset = 60
                for action, score in predictions.items():
                    print(f'{action}: {score:.2f}')

            cv2.imshow('Classroom Monitoring', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

run_multiple_inference()