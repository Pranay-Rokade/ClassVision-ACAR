import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import defaultdict

# Load YOLO and LRCN model
yolo_model = YOLO("yolo11n.pt")  # Replace with your YOLO model
pose_model = load_model("without_sleeping.h5")  

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()




####################################################################################
actions = np.array(['Eating_in_classroom', 'Ha"Analyzing"ndRaise', 'Reading_Book','Sitting_on_Desk','Writting_on_Textbook',"Analyzing"])


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh

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
####################################################################################



# Constants
SEQUENCE_LENGTH = 30
FRAME_SIZE = (255, 255)
UNKNOWN_ACTIVITY = 5
FRAME_INTERVAL = 6  # Process every 6th frame for video files

# Tracking data for students
student_sequences = defaultdict(lambda: [])
student_activities = {}



# Function to predict posture
def predict_posture(sequence):
    sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
    prediction = pose_model.predict(sequence)
    return np.argmax(prediction)

# Function for multi-student inference
def run_multi_student_inference(video_path=None):
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    frame_count = 0  # Track frame number
    with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            print(frame_count)
            # Process frames at intervals if video file
            if frame_count % FRAME_INTERVAL != 0:
                frame_count += 1
                print("continue")
                continue

            print("after_continue",frame_count)

            # YOLO Detection
            results = yolo_model.track(frame, persist=True, classes=[0])  # Detect only humans
            detections = results[0].boxes.data.cpu().numpy() if results[0].boxes else []

            for box in detections:
                x1, y1, x2, y2, track_id, _ = map(int, box[:6])
                roi = frame[y1:y2, x1:x2]
                roi = cv2.resize(roi, FRAME_SIZE)

                ####################################################################################
                image, mediapip_results = mediapipe_detection(roi, holistic)
                # draw_landmarks(frame, mediapip_results)


                #Prediction Logic
                keypoints = extract_keypoints(mediapip_results)
                student_sequences[track_id].append(keypoints)

                if len(student_sequences[track_id]) > SEQUENCE_LENGTH:
                    student_sequences[track_id] = student_sequences[track_id][15:]  # Removes first 15 elements

                
                if len(student_sequences[track_id]) == SEQUENCE_LENGTH:
                    predicted_label_idx = predict_posture(np.array(student_sequences[track_id]))
                    student_activities[track_id] = predicted_label_idx
                else:
                    
                    
                    student_activities[track_id] = UNKNOWN_ACTIVITY
                
                # Draw bounding box & label
                
                activity_label = actions[student_activities.get(track_id, UNKNOWN_ACTIVITY)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Posture: {activity_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # if pose_landmarks:
                #     mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            cv2.imshow("ClassVision - Multi-Student Pose Detection", frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Run multi-student inference
run_multi_student_inference()
