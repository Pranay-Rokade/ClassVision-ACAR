from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import StreamingHttpResponse, FileResponse, JsonResponse, HttpResponse
from django.core.files.storage import default_storage
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
import requests
import psutil
import os
from datetime import datetime
import face_recognition
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque, defaultdict
from tensorflow.keras.models import load_model
import subprocess
import shutil
import pickle
import csv
from analysis.models import ClassData   
from django.core.files import File
from django.core.files.storage import FileSystemStorage
import threading

# For Face Recognition of students for logging purpose
def face_encodings_for_dataset(image_data_path = "/run/media/yashtapre77/New Volume/Projects/Class Vision/backend/classification/images"):
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



# Create your views here.
VIDEO_URL = None  # Placeholder for the video URL

# mediapipe model initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh

print(os.getcwd())
# Model initialization
os.chdir('classification/weights')
model = load_model("actionsIncludingSleeping.h5")
yolo_model = YOLO("yolo11n.pt")
os.chdir('../..')
print(os.getcwd())


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
FRAME_INTERVAL = 6
RESIZE_WIDTH = 224
RESIZE_HEIGHT = 224
SEQUENCE_LENGTH = 30
MODEL_CALL_INTERVAL = 1
bbox_history = defaultdict(lambda: [])
sequence = []
students_sequences = defaultdict(lambda: [])  # stores keypoints for each student
last_action = defaultdict(lambda: [])
last_bounding_box = defaultdict(lambda: [])

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

def is_file_in_use(file_path):
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if file_path in item.path:
                    return True
        except Exception:
            pass
    return False

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

def schedule_folder_cleanup(folder_paths, delay=3):
    def delete_folder():
        try:
            for folder_path in folder_paths:
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                    print(f"Deleted temp folder: {folder_path}")
                print("for")
            print("try")
        except Exception as e:
            print(f"Error deleting folder: {folder_path} -> {e}")
    print(f"Scheduling folder cleanup in {delay} seconds...")
    threading.Timer(delay, delete_folder).start()

def process_video(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    FRAME_COUNT = 0
    file_path = os.path.join(default_storage.location, "logs/data.csv")
    header_row = ["Name", "Action", "Timestamp"]

    # temporary file, so that every entry will rewrite the previous one, only single file will be maintained
    f = open(file_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(header_row)
    frame_index = 0
    


    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    # Define output path for processed video
    output_path = os.path.join(default_storage.location,"processed_videos", f"analyzed{ext}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print("*****************************************************************************************************************************************")
    print()
    print()
    print(fps)
    print()
    print()
    print("*****************************************************************************************************************************************")

    # for i in encodings.keys():
    #     writer.writerow([i, "Siting on Desk", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])   


    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_index += 1
            if FRAME_COUNT % FRAME_INTERVAL != 0:
                for i in last_bounding_box.keys():
                    x1, y1, x2, y2 = last_bounding_box[i]
                    if last_action[i] in PRODUCTIVE_ACTIONS:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, last_action[i], (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else :
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, last_action[i], (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                out.write(frame)
                FRAME_COUNT += 1
                continue


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

                        # detect student face in video and log their action as using phone
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
                        last_bounding_box[closest_person_id] = (x1, y1, x2, y2)

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
                    last_bounding_box[track_id] = (x1, y1, x2, y2)

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
                name = "Unknown"
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(list(encodings.values()), face_encoding)
                    face_distances = face_recognition.face_distance(list(encodings.values()), face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = list(encodings.keys())[best_match_index]
                    
                if last_act != predicted_action:
                    print(f"Detected: {name} - Action: {predicted_action}")
                    writer.writerow([name, predicted_action, f"{formatted_date} {timestamp_str}"])

            FRAME_COUNT += 1
            out.write(frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_path, file_path



# Receive Video URL Class
class live_video(APIView):
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




class upload_video(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            # Get video file from request
            video_file = request.FILES.get("video")

            if not video_file:
                return Response({"error": "No video file provided"}, status=400)
            
            # Save the uploaded video temporarily
            ext = os.path.splitext(video_file.name)[1]

            temp_video_path = os.path.join("temp", f"tempfile{ext}")

            # Delete the file if it already exists
            if default_storage.exists(temp_video_path):
                default_storage.delete(temp_video_path)

            # Now save the uploaded file
            video_path = default_storage.save(temp_video_path, video_file)

            video_full_path = os.path.join(default_storage.location, video_path)

            # Process the video
            processed_video_path, file_path = process_video(video_full_path)

            convert_to_streamable_mp4(processed_video_path, "media/processed_videos/output.mp4")

            output_video_path = os.path.join(default_storage.location,"processed_videos", "output.mp4")
            # Send the processed video back to React
            response = FileResponse(open(output_video_path, "rb"), content_type="video/mp4", status=200)
            response["Content-Disposition"] = f'attachment; filename="processed_video.mp4"'

            
            # time.sleep(1)
            # if not is_file_in_use(processed_video_path):
            #     os.remove(processed_video_path)
            #     print("file removed")
            # else:
            #     print(f"Skipping deletion: {processed_video_path} is still in use.")

            # Clean up temporary files
            # os.remove(video_full_path)
            # os.remove(processed_video_path)
            # os.remove(output_video_path)

            class_name = "Class A"  # Get this from request or context

            with open(file_path, 'rb') as f:
                django_file = File(f)

                try:
                    # Try to get the existing record
                    class_data = ClassData.objects.get(class_name=class_name)

                    # Delete the old file if it exists
                    if class_data.csv_file:
                        class_data.csv_file.delete(save=False)

                    # Save the new file with the same name
                    class_data.csv_file.save("log_class_a.csv", django_file, save=True)
                    class_data.description = "Updated after re-processing"
                    class_data.mode = "student"  # or other value
                    class_data.save()
                except ClassData.DoesNotExist:
                    # Create a new record
                    ClassData.objects.create(
                        class_name=class_name,
                        csv_file=django_file,
                        description="Created after processing",
                        mode="student"
                    )


            # Schedule folder cleanup
            # paths = [os.path.join(default_storage.location, "temp"), os.path.join(default_storage.location, "processed_videos")]
            # schedule_folder_cleanup(paths, delay=500)

            return response

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

