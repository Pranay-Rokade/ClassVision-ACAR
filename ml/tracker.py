import cv2
import os
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh

os.chdir('weights')
model = load_model("actionsIncludingSleeping.h5")
os.chdir('..')

yolo = YOLO("yolo11n.pt")

bbox_history = deque(maxlen=10)

actions = ['Eating in Classroom', 'Hand Raise', 'Reading Book', 'Sitting on Desk', 'Sleeping', 'Writing in Textbook']

label_map = {'Eating in Classroom': 0,
 'Hand Raise': 1,
 'Reading Book': 2,
 'Sitting on Desk': 3,
 'Sleeping': 4,
 'Writing in Textbook': 5}


sequence = []
sentence = []
threshold = 0.4

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


sequence = []
sentence = []
threshold = 0.4
cap = cv2.VideoCapture(0)
# prev_results = {}

with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, static_image_mode = False) as holistic:
    while cap.isOpened():
        _, frame = cap.read()
        results = yolo(frame)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
            cls = result.boxes.cls.cpu().numpy()  # Get class IDs

            for i, box in enumerate(boxes):
                if int(cls[i]) == 0:  # Only consider 'person' class (ID 0)
                    x1, y1, x2, y2 = map(int, box)

                    # Apply bounding box smoothing
                    x1, y1, x2, y2 = smooth_bbox((x1, y1, x2, y2))

                    # Expand bounding box for stability
                    x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, scale=1.2)

                    # Crop and resize
                    person_crop = frame[y1:y2, x1:x2]
                    person_crop_resized = cv2.resize(person_crop, (600, 600))  # Resize to your model's input size
                    image, results = mediapipe_detection(person_crop_resized, holistic)
                    draw_landmarks(person_crop_resized, results)
                    #Prediction Logic
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence)==30:
                        res = model.predict(np.expand_dims(sequence, axis=0))
                        res = np.squeeze(res)  # Ensures it's a 1D array

                        # Display each action with its probability
                        for idx, prob in enumerate(res):
                            text = f"{actions[idx]}: {prob:.2f}"  # No need for float(), NumPy float64 supports formatting
                            cv2.putText(person_crop_resized, text, (10, 50 + (idx * 30)), cv2.FONT_HERSHEY_COMPLEX, 
                                        0.6, (0, 0, 0), 2)

                        # Display predicted action with the highest probability
                        predicted_action = actions[np.argmax(res)]
                        # prev_results[track_id] = predicted_action
                        cv2.putText(person_crop_resized, f"Pred: {predicted_action}", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('OpenCV Feed', person_crop_resized)


        # cv2.imshow('OpenCV Feed', frame)

        if(cv2.waitKey(10) & 0xFF == ord('q')):
            break
cap.release()
cv2.destroyAllWindows()