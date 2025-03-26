import cv2
import torch
import numpy as np
from ultralytics import YOLO

def get_center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def detect_phone_usage(video_path=0):
    model = YOLO("yolo11n.pt")  # Load YOLOv8 model
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        phones = []
        persons = []
        
        for result in results:
            for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                if int(cls) == 67:  # Class ID 67 corresponds to 'cell phone' in COCO
                    phones.append(box)
                elif int(cls) == 0:  # Class ID 0 corresponds to 'person'
                    persons.append(box)
        
        if phones and persons:
            for phone in phones:
                phone_center = get_center(phone)
                
                min_distance = float('inf')
                closest_person = None
                phone_coordinates = phone
                
                for person in persons:
                    person_center = get_center(person)
                    distance = abs(phone_center[0] - person_center[0]) + abs(phone_center[1] - person_center[1])
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_person = person
                
                if closest_person is not None:
                    x1, y1, x2, y2 = map(int, closest_person)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Using Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    x1, y1, x2, y2 = map(int, phone_coordinates)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.imshow("Phone Usage Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

detect_phone_usage(0)  # Use webcam (0) or provide a video path
