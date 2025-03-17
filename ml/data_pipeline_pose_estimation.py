import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


#The height and width of each frame in the video
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

#no of frame that will used from each video
SEQUENCE_LENGTH = 20

CLASS_LIST = ['Eating in Classroom','Hand Raise','Sitting on Desk']
NUM_CLASSES = len(CLASS_LIST)

#Student body max key points
NUM_KEYPOINTS = 33

#values for each keypoint x,y,z,visibility
FEATURES_PER_KEYPOINT = 4

#input shape for model
INPUT_SHAPE = (SEQUENCE_LENGTH, NUM_KEYPOINTS, FEATURES_PER_KEYPOINT)

# Process every 6th frame in video file
FRAME_INTERVAL = 6

MODEL = load_model("LRCN_model___Date_Time_2025_03_14__05_50_25___Loss_0.5764392018318176___Accuracy_0.8064516186714172.h5")


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()


def extract_keypoints(frame):
    """Extracts keypoints from the given frame using Mediapipe Pose Detection."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    keypoints = np.zeros((NUM_KEYPOINTS, FEATURES_PER_KEYPOINT))
    pose_landmarks = None  # Default to None
    
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks  # Assign pose landmarks
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
    
    return keypoints, pose_landmarks  # Ensure we return two values


def predict_posture(sequence):
    sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
    prediction = MODEL.predict(sequence)
    return np.argmax(prediction)




def run_inference(video_path=None):
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    sequence_buffer = []
    frame_count = 0  # Track frame number
    predicted_label = "Analyzing..."  # Default label

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        # Process every 6th frame if video file
        if video_path and frame_count % FRAME_INTERVAL != 0:
            frame_count += 1
            continue

        # Extract keypoints
        keypoints, pose_landmarks = extract_keypoints(frame)  # Implement this function
        sequence_buffer.append(keypoints)

        # Maintain a fixed sequence length of 20
        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)

        # If we have enough frames, predict posture
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            predicted_label_idx = predict_posture(np.array(sequence_buffer))
            predicted_label = CLASS_LIST[int(predicted_label_idx)]
            # print(f"Predicted Posture: {predicted_label}")
            # print(type(predicted_label))
            # print(CLASS_LIST[int(predicted_label)])

        if pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)

        
        # Display predicted posture on the frame
        cv2.putText(
            frame, 
            f"Posture: {predicted_label}", 
            (30, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2, cv2.LINE_AA
        )

        # Show the frame
        cv2.imshow("Posture Detection", frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# run_inference("http://192.0.0.4:8080/video")
run_inference()