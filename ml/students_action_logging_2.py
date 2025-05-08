import cv2
import numpy as np
from datetime import datetime 
import csv
import os
import face_recognition
import pickle

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

encodings = face_encodings_for_dataset()
print(encodings)
