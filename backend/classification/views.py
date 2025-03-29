from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import StreamingHttpResponse, FileResponse, JsonResponse, HttpResponse
from django.core.files.storage import default_storage
from rest_framework.parsers import MultiPartParser, FormParser
import cv2
import os
from django.conf import settings
import subprocess
import tempfile
import cv2
import os
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque, defaultdict
from tensorflow.keras.models import load_model
import time
import torch
import threading
from queue import Queue
import concurrent.futures

# Create your views here.
VIDEO_URL = None  # Placeholder for the video URL


# Receive Video URL Class
class receive_video_url(APIView):
    def post(self, request):  
        try:
            # Get the video URL from the request
            video_url = request.data.get("videourl")
            # video_url = "https://192.0.0.4:8080/video"

            if not video_url:
                return Response({"error": "No video URL provided"}, status=400)

            # Validate URL
            if not video_url.startswith("http"):
                return Response({"error": "Invalid video URL"}, status=400)

            # Stream the video for real-time processing
            return StreamingHttpResponse(
                generate_frames(video_url),
                content_type="multipart/x-mixed-replace;boundary=frame",
                status=200
            )

        except Exception as e:
            return Response({"error": str(e)}, status=500)
        
    def get(self, request):  
        try:
            # Get the video URL from the request
            # video_url = request.data.get("videourl")
            video_url = "https://192.0.0.4:8080/video"

            if not video_url:
                return Response({"error": "No video URL provided"}, status=400)

            # Validate URL
            if not video_url.startswith("http"):
                return Response({"error": "Invalid video URL"}, status=400)

            # Stream the video for real-time processing
            return StreamingHttpResponse(
                generate_frames(video_url),
                content_type="multipart/x-mixed-replace;boundary=frame",
                status=200
            )

        except Exception as e:
            return Response({"error": str(e)}, status=500)


# Video frame generator for live video processing
def generate_frames(video_url):
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        yield b""

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Process frame (Add your custom analysis logic here)
        frame = process_frame(frame)

        # Encode and yield the frame
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


# Example function to process the frame
def process_frame(frame):
    # Add any frame processing logic here (e.g., object detection, pose estimation, etc.)
    cv2.putText(frame, "Processing...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


class VideoClassification(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        video_full_path = None
        processed_video_path = None
        file_handle = None
        
        try:
            # Get video file from request
            video_file = request.FILES.get("video")
            
            if not video_file:
                return Response({"error": "No video file provided"}, status=400)
            
            # Save the uploaded video temporarily
            video_path = default_storage.save("temp/" + video_file.name, video_file)
            video_full_path = os.path.join(default_storage.location, video_path)
            
            # Process the video
            processed_video_path = process_video(video_full_path)
            # processed_video_path = os.path.join(default_storage.location,"processed_videos", "plane.mp4")
            print(f"Processed video path: {processed_video_path}")
            # Ensure the file exists and is readable
            if not os.path.exists(processed_video_path):
                return Response({"error": "Failed to create processed video"}, status=500)
            
            # Determine content type based on file extension
            content_type = "video/mp4" if processed_video_path.endswith(".mp4") else "video/x-msvideo"
            file_extension = os.path.splitext(processed_video_path)[1]
            
            # Create the response - but don't use 'with' statement here
            # The FileResponse will handle closing the file when done
            file_handle = open(processed_video_path, "rb")
            response = FileResponse(file_handle, content_type=content_type, status=200)
            response["Content-Disposition"] = f'attachment; filename="processed_video{file_extension}"'
            
            # Add CORS headers
            response["Access-Control-Allow-Origin"] = "*"  # Or your specific domain
            response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response["Access-Control-Allow-Headers"] = "Content-Type"
            

            file_handle = None
            
            # Clean up the input file - we can do this safely now
            if video_full_path and os.path.exists(video_full_path):
                try:
                    os.remove(video_full_path)
                except:
                    pass
            video_full_path = None
            
            return response
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error processing video: {error_details}")
            return JsonResponse({"error": str(e)}, status=500)
        
        finally:
            # Clean up the input file if it still exists
            if video_full_path and os.path.exists(video_full_path):
                try:
                    os.remove(video_full_path)
                except:
                    pass

# Load Models
# os.chdir('..')
os.chdir('classification')
os.chdir('weights')
model = load_model("actionsIncludingSleeping.h5")
yolo_model = YOLO("yolo11n.pt").to("cuda" if torch.cuda.is_available() else "cpu")
os.chdir('..')
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
student_results = {}  # Store processing results for each student

# Thread lock for shared resources
lock = threading.Lock()

SEQUENCE_LENGTH = 30
MODEL_CALL_INTERVAL = 1
FRAME_INTERVAL = 0
THRESHOLD = 0.10

import time
timestamp = int(time.time() * 1e6)  # Get current timestamp in microseconds


## ----- HELPER FUNCTIONS ----- ###

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
    with lock:
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



### ----- STUDENT PROCESSING FUNCTION FOR THREADING ----- ###

def process_student(track_id, person_box, frame, holistic):
    """Process a single student in a separate thread"""
    # Extract and normalize coordinates
    x1, y1, x2, y2 = map(int, person_box)
    h, w, _ = frame.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    
    # Apply bounding box smoothing and expansion
    x1, y1, x2, y2 = smooth_bbox(track_id, (x1, y1, x2, y2))
    x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, frame, scale=1.2)
    
    # Extract ROI
    roi = frame[y1:y2, x1:x2]
    if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
        return None
    
    person_crop_resized = cv2.resize(roi, (224, 224))
    
    # Extract keypoints
    image, results = mediapipe_detection(person_crop_resized, holistic)
    keypoints = extract_keypoints(results)
    
    # Update sequence data with thread safety
    with lock:
        student_sequences[track_id].append(keypoints)
        
        if track_id not in sequence_data:
            sequence_data[track_id] = []
        sequence_data[track_id].append(keypoints)
        
        if len(sequence_data[track_id]) > SEQUENCE_LENGTH:
            sequence_data[track_id] = sequence_data[track_id][MODEL_CALL_INTERVAL:]
    
    predicted_action = "Sitting on Desk"
    
    # If we have enough frames, predict the action
    with lock:
        if len(sequence_data[track_id]) == SEQUENCE_LENGTH:
            res = model.predict(np.expand_dims(sequence_data[track_id], axis=0))
            res = np.squeeze(res)
            predicted_action = actions[np.argmax(res)]
            confidence = res[np.argmax(res)]
            
            if confidence > THRESHOLD:
                last_action[track_id] = predicted_action
            else:
                last_action[track_id] = "Sitting on Desk"
    
    # Return the processing results
    return {
        'track_id': track_id,
        'bbox': (x1, y1, x2, y2),
        'action': predicted_action,
        'is_productive': predicted_action in PRODUCTIVE_ACTIONS
    }

### ----- VIDEO PROCESSING ----- ###

def process_video(video_path=None):
    # Set up video capture
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    
    # Set up video writer if video_path is provided
    output_video = None
    if video_path is not None:
        base_name = os.path.basename(video_path)
        name, ext = os.path.splitext(base_name)
        output_path = f"{name}_analyzed{ext}"
        output_path = os.path.join(default_storage.location, "processed_videos", output_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    # Create a thread pool for processing students
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor, \
         mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            # Detect objects using YOLO
            detected_objects = yolo_model.track(frame, persist=True, classes=[0, 67])  # Detect only humans and phones
            
            phones = []
            persons = {}
            person_ids_using_phone = []
            
            # Extract detection results
            for obj in detected_objects:
                if obj.boxes.id is None:  # Skip if no tracking ID assigned
                    continue
                for box, cls, track_id in zip(obj.boxes.xyxy.cpu().numpy(), obj.boxes.cls.cpu().numpy(), obj.boxes.id):
                    if int(cls) == 67:  # 'cell phone'
                        phones.append(box)
                    elif int(cls) == 0 and track_id is not None:  # 'person' with ID
                        persons[int(track_id)] = box
            
            # Match phones to persons
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
                        
                        # Draw bounding box around phone user
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Using Phone (ID {closest_person_id})", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Draw phone bounding box
                        x1, y1, x2, y2 = map(int, phone)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Process each person in a separate thread
            future_to_student = {}
            for track_id, person_box in persons.items():
                # Skip processing for students identified as using phones
                if track_id in person_ids_using_phone:
                    continue
                
                # Submit the task to the thread pool
                future = executor.submit(process_student, track_id, person_box, frame.copy(), holistic)
                future_to_student[future] = track_id
            
            # Collect results from completed threads
            student_results.clear()
            for future in concurrent.futures.as_completed(future_to_student):
                result = future.result()
                if result:
                    student_results[result['track_id']] = result
            
            # Draw results on the frame
            for phone_user_id in person_ids_using_phone:
                if phone_user_id in persons:
                    x1, y1, x2, y2 = map(int, persons[phone_user_id])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Using Phone (ID {phone_user_id})", 
                               (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            for track_id, result in student_results.items():
                x1, y1, x2, y2 = result['bbox']
                action = result['action']
                is_productive = result['is_productive']
                
                # Draw bounding box and action text
                color = (0, 255, 0) if is_productive else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, action, (x1 + 10, y1 + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # Write to output video if saving
            if output_video is not None:
                output_video.write(frame)
            
            # Display the frame
            # cv2.imshow('Classroom Monitoring', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            frame_count += 1
    
    # Release resources
    cap.release()
    if output_video is not None:
        output_video.release()
    cv2.destroyAllWindows()

    return output_path