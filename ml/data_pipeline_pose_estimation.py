import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import defaultdict

# Load YOLO and LRCN model
yolo_model = YOLO("yolo11n.pt")  # Replace with your YOLO model
pose_model = load_model("LRCN_model___Date_Time_2025_03_14__05_50_25___Loss_0.5764392018318176___Accuracy_0.8064516186714172.h5")  

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Constants
SEQUENCE_LENGTH = 20
FRAME_SIZE = (64, 64)
UNKNOWN_ACTIVITY = "Analyzing..."
FRAME_INTERVAL = 6  # Process every 6th frame for video files

# Tracking data for students
student_sequences = defaultdict(lambda: [])
student_activities = {}

# Function to extract pose keypoints
def extract_keypoints(roi):
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = pose.process(roi_rgb)
    keypoints = np.zeros((33, 4))  # 33 keypoints, each with x, y, z, visibility
    
    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
    
    return keypoints, results.pose_landmarks

# Function to predict posture
def predict_posture(sequence):
    sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
    prediction = pose_model.predict(sequence)
    return np.argmax(prediction)

# Function for multi-student inference
def run_multi_student_inference(video_path=None):
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    frame_count = 0  # Track frame number

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frames at intervals if video file
        if video_path and frame_count % FRAME_INTERVAL != 0:
            frame_count += 1
            continue

        # YOLO Detection
        results = yolo_model.track(frame, persist=True, classes=[0])  # Detect only humans
        detections = results[0].boxes.data.cpu().numpy() if results[0].boxes else []

        for box in detections:
            x1, y1, x2, y2, track_id, _ = map(int, box[:6])
            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, FRAME_SIZE)

            keypoints, pose_landmarks = extract_keypoints(roi)
            student_sequences[track_id].append(keypoints)
            
            if len(student_sequences[track_id]) > SEQUENCE_LENGTH:
                student_sequences[track_id].pop(0)
            
            if len(student_sequences[track_id]) == SEQUENCE_LENGTH:
                predicted_label_idx = predict_posture(np.array(student_sequences[track_id]))
                student_activities[track_id] = predicted_label_idx
            else:
                student_activities[track_id] = UNKNOWN_ACTIVITY
            
            # Draw bounding box & label
            activity_label = str(student_activities.get(track_id, UNKNOWN_ACTIVITY))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Posture: {activity_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow("ClassVision - Multi-Student Pose Detection", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run multi-student inference
run_multi_student_inference()
