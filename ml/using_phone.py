import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO

def get_center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def detect_phone_usage(video_path=0):
    model = YOLO("yolo11n.pt").to("cuda" if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)  # Use YOLO's built-in tracking
        phones = []
        persons = {}

        for result in results:
            if result.boxes.id is None:  # Skip if no tracking ID assigned
                continue
            for box, cls, track_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.id):
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
                    x1, y1, x2, y2 = map(int, persons[closest_person_id])

                    # Draw bounding box only around phone user
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Using Phone (ID {closest_person_id})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    x1, y1, x2, y2 = map(int, phone)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

   

        cv2.imshow("Phone Usage Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_phone_usage(0)  # Use webcam (0) or provide a video path
