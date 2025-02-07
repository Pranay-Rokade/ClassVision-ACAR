import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque

# Load the trained LRCN model
model = load_model("LRCN_model___Date_Time_2025_02_07__02_48_19___Loss_0.15253400802612305___Accuracy_0.9473684430122375.h5")  # Replace with actual model path

# Define frame preprocessing function
def preprocess_frame(frame):
    img_size = (64, 64)  # Change this based on your model's expected input
    frame = cv2.resize(frame, img_size)  # Resize frame
    frame = frame / 255.0  # Normalize pixel values
    return frame

# Initialize the frame buffer (deque for storing last 20 frames)
SEQUENCE_LENGTH = 20
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Preprocess frame and add to buffer
    processed_frame = preprocess_frame(frame)
    frame_buffer.append(processed_frame)

    # Only predict when we have 20 frames
    if len(frame_buffer) == SEQUENCE_LENGTH:
        input_sequence = np.expand_dims(np.array(frame_buffer), axis=0)  # Add batch dimension

        # Predict action
        prediction = model.predict(input_sequence)
        predicted_class = np.argmax(prediction)

        # Map class index to action label
        class_labels = ["Eating", "Sitting on desk", "Hand Raising", "Reading", "Writing", "Talking", "Other"]
        action = class_labels[predicted_class]

        # Display prediction
        cv2.putText(frame, f"Action: {action}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("Live Action Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
