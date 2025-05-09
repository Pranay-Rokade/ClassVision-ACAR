import cv2
import numpy as np
from datetime import datetime 
import csv
import os
import face_recognition
import pickle
import mediapipe as mp
from ultralytics import YOLO
from collections import deque, defaultdict
from tensorflow.keras.models import load_model
import subprocess

def face_encodings_for_dataset(image_data_path = "images"):
    encodings = {}
    for filename in os.listdir(image_data_path):
        name = os.path.splitext(filename)[0]
        print(f"Processing {filename}, {name}")
        img = face_recognition.load_image_file(os.path.join(image_data_path, filename))
        encodings[name] = face_recognition.face_encodings(img)[0]
    
    with open('encodings.pickle', 'wb') as f:
        pickle.dump(encodings, f)

    return encodings

if os.path.isfile('encodings.pickle'):
    with open('encodings.pickle', 'rb') as f:
        encodings = pickle.load(f)
else:
    encodings = face_encodings_for_dataset() 

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H-%M-%S")







# mediapipe model initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh

# Model initialization
os.chdir('weights')
model = load_model("actionsIncludingSleeping.h5")
yolo_model = YOLO("yolo11n.pt")
os.chdir('..')


# Actions and labels to be detected
ACTIONS = ['Eating in Classroom', 'Hand Raise', 'Reading Book', 'Sitting on Desk', 'Sleeping', 'Writing in Textbook', "Using Phone"]

DISTRACTED_ACTIONS = ['Eating in Classroom', 'Sleeping', 'Using Phone']
PRODUCTIVE_ACTIONS = ['Hand Raise', 'Reading Book', 'Sitting on Desk', 'Writing in Textbook']

label_map = {'Eating in Classroom': 0,
 'Hand Raise': 1,
 'Reading Book': 2,
 'Sitting on Desk': 3,
 'Sleeping': 4,
 'Writing in Textbook': 5,
 'Using Phone': 6}


# Initialize variables and Constants
THRESHOLD = 0.4
FRAME_INTERVAL = 1
RESIZE_WIDTH = 224
RESIZE_HEIGHT = 224
SEQUENCE_LENGTH = 30
MODEL_CALL_INTERVAL = 1
bbox_history = defaultdict(lambda: [])
sequence = []
students_sequences = defaultdict(lambda: [])  # stores keypoints for each student
last_action = defaultdict(lambda: [])


# Helper Functions for frame processing and normalization
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def smooth_bbox(track_id, new_bbox):
    """Smooth bounding box by averaging recent detections."""
    if track_id not in bbox_history:
        bbox_history[track_id] = []
    bbox_history[track_id].append(new_bbox)
    avg_bbox = np.mean(bbox_history[track_id], axis=0)  # Compute the average bounding box
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

def get_center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))



# Function to classify the action
def run_multiple_inference(video_path=None):
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    FRAME_COUNT = 0

    f = open(f"{current_date}_{current_time}.csv", 'w', newline='')
    writer = csv.writer(f)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    output_path = None
    # save video only if video_path of video is provided, if webcam or IP is provided then no need to save, just show live
    output_video = None
    if video_path is not None and video_path.endswith(('.mp4', '.avi', '.mov')):
        base_name = os.path.basename(video_path)
        name, ext = os.path.splitext(base_name)
        output_path = f"{name}_analyzed{ext}"
         
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
         
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    
    for i in encodings.keys():
        writer.writerow([i, "Siting on Desk", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])    

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            frame_index += 1
            # if frame_count % FRAME_INTERVAL != 0:
            #     frame_count += 1
            #     continue

            # Calculate timestamp in seconds
            timestamp_sec = frame_index / fps

            # Optionally, convert to HH:MM:SS format
            hours = int(timestamp_sec // 3600)
            minutes = int((timestamp_sec % 3600) // 60)
            seconds = int(timestamp_sec % 60)
            millis = int((timestamp_sec % 1) * 1000)

            timestamp_str = f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"
            formatted_date = now.strftime("%Y-%m-%d") 

            detected_obejets = yolo_model.track(frame, persist=True, classes=[0, 67])  # Detect only humans and phones

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

                        roi = frame[y1:y2, x1:x2]
                        if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                            continue
                        person_crop_resized = cv2.resize(roi, (RESIZE_WIDTH, RESIZE_HEIGHT))  
                        rgb_small_frame = cv2.cvtColor(person_crop_resized, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_small_frame)
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                        # print(face_encodings)

                        for face_encoding in face_encodings:
                            matches = face_recognition.compare_faces(list(encodings.values()), face_encoding)
                            face_distances = face_recognition.face_distance(list(encodings.values()), face_encoding)
                            best_match_index = np.argmin(face_distances)
                            name = "Unknown"
                            if matches[best_match_index]:
                                name = list(encodings.keys())[best_match_index]
                    
                            if last_action[closest_person_id] != "Using Phone":
                                print(f"Detected: {name} - Action: {predicted_action}")
                                writer.writerow([name, "Using Phone", f"{formatted_date} {timestamp_str}"])

                        last_action[closest_person_id] = "Using Phone"  # Store last detected action
                        # Draw bounding box only around phone user
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Using Phone (ID {closest_person_id})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        x1, y1, x2, y2 = map(int, phone)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            
            for track, person in persons.items():

                # if person is identified as using phone then skip his action classification
                if track in person_ids_using_phone:
                    continue

                # track id and area extraction of each person
                track_id = track
                x1, y1, x2, y2 = map(int, person)
                h, w, _ = frame.shape
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                # cropped area normalizaion of each person
                x1, y1, x2, y2 = smooth_bbox(track_id, (x1, y1, x2, y2))
                x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, scale=1.2)

                # region of interest of each person
                roi = frame[y1:y2, x1:x2]
                if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                        continue
                person_crop_resized = cv2.resize(roi, (RESIZE_WIDTH, RESIZE_HEIGHT))  


                # keypoints extraction of each cropped person
                _ , results = mediapipe_detection(person_crop_resized, holistic)
                # draw_landmarks(person_crop_resized, results)
                keypoints = extract_keypoints(results)

                
                predicted_action = "Sitting on Desk"   # Default action if, predicted action does not exceed threshold


                # to keep track of each person's keypoints and predictions
                if track_id not in students_sequences:
                    students_sequences[track_id] = []
                students_sequences[track_id].append(keypoints)


                if len(students_sequences[track_id]) > SEQUENCE_LENGTH:
                    students_sequences[track_id] = students_sequences[track_id][MODEL_CALL_INTERVAL:]  # Remove first MODEO_CALL_INTERVAL frames

                last_act = "Sitting on Desk"  # Default action if, predicted action does not exceed threshold
                if len(students_sequences[track_id]) == SEQUENCE_LENGTH:
                    res = model.predict(np.expand_dims(students_sequences[track_id], axis=0))
                    res = np.squeeze(res)
                    predicted_action = ACTIONS[np.argmax(res)]
                    confidence = res[np.argmax(res)]

                    last_act = last_action[track_id]

                    if confidence > THRESHOLD:  # Confidence threshold
                        last_action[track_id] = predicted_action  # Store last detected action
                    else:
                        last_action[track_id] = "Sitting on Desk"

                if predicted_action in PRODUCTIVE_ACTIONS:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, predicted_action, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else :
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, predicted_action, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # face detection and encoding
                # small_frame = cv2.resize(person_crop_resized, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(person_crop_resized, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                # print(face_encodings)

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(list(encodings.values()), face_encoding)
                    face_distances = face_recognition.face_distance(list(encodings.values()), face_encoding)
                    best_match_index = np.argmin(face_distances)
                    name = "Unknown"
                    if matches[best_match_index]:
                        name = list(encodings.keys())[best_match_index]
                    
                    if last_act != predicted_action:
                        print(f"Detected: {name} - Action: {predicted_action}")
                        writer.writerow([name, predicted_action, f"{formatted_date} {timestamp_str}"])

                print(f"Detected")
            if output_video is not None:
                 output_video.write(frame)
            else:
                cv2.imshow('Classroom Monitoring', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    if output_video is not None:
         output_video.release()
    cap.release()
    f.close()
    cv2.destroyAllWindows()
    return output_path   


def convert_to_streamable_mp4(input_path, output_path):
    command = [
        "ffmpeg",
        "-y",  # overwrite if exists
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(command, check=True)
    


output_path = run_multiple_inference("test3.mp4")
convert_to_streamable_mp4(output_path, "final_output.mp4")
