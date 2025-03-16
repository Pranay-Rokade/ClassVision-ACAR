import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import defaultdict

# Load YOLO and LRCN model
yolo_model = YOLO("yolo11n.pt")  # Replace with your trained YOLO model
lrcn_model = load_model("LRCN_demo.h5")  # Replace with your trained LRCN model

# Constants
SEQUENCE_LENGTH = 20
FRAME_SIZE = (64, 64)
UNKNOWN_ACTIVITY = "Unknown"
CLASS_LIST = ["Eating in Classroom","Sitting on Desk", "Hand Raising"]
# VIDEO_URL = "http://192.0.0.4:8080/video"  # to get live footage from mobile camera
VIDEO_URL = 0  # To get live footage from laptop's webcam


# Initialize tracking data
student_sequences = defaultdict(lambda: [])  # Stores sequences of 20 frames for each student
student_activities = {}  # Stores latest predicted activities

# Capture live video
cap = cv2.VideoCapture(VIDEO_URL)  # Change to file path if using a video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Detection
    results = yolo_model.track(frame, persist=True, classes=[0])  # Detect only humans
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes else []

    for box in detections:
        x1, y1, x2, y2, track_id, _ = map(int, box[:6])

        # Extract ROI and resize to LRCN input size
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, FRAME_SIZE)
        roi = roi / 255.0  # Normalize

        # Maintain sequence of 20 frames for each student
        student_sequences[track_id].append(roi)
        if len(student_sequences[track_id]) > SEQUENCE_LENGTH:
            student_sequences[track_id].pop(0)  # Keep only last 20 frames

        # Predict activity if we have 20 frames
        if len(student_sequences[track_id]) == SEQUENCE_LENGTH:
            input_sequence = np.expand_dims(np.array(student_sequences[track_id]), axis=0)  # Shape: (1, 20, 64, 64, 3)
            prediction = lrcn_model.predict(input_sequence)
            student_activities[track_id] = CLASS_LIST[np.argmax(prediction)]

        else:
            student_activities[track_id] = UNKNOWN_ACTIVITY  # Show 0 (Unknown) until LRCN predicts

        # Draw bounding box and activity label
        activity_label = f"{student_activities.get(track_id, UNKNOWN_ACTIVITY)}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, activity_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display output frame
    cv2.imshow("ClassVision - Live Activity Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
