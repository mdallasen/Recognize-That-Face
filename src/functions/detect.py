import os
import sys
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from retinaface import RetinaFace
from tensorflow.keras.preprocessing.image import img_to_array

def detect_label(dataset_path): 

    data = [] 
    size = (160, 160)

    for label in os.listdir(dataset_path):  
        label_path = os.path.join(dataset_path, label)

        if os.path.isdir(label_path):  
            for image_file in os.listdir(label_path):  
                image_path = os.path.join(label_path, image_file)
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Warning: Unable to read {image_path}")
                    continue  
                
                faces = RetinaFace.detect_faces(img) if img is not None else None  

                if faces is None or len(faces) == 0: 
                    print(f"Warning: No face detected in {image_path}")
                    continue

                for _, face_info in faces.items():
                    x1, y1, x2, y2 = face_info["facial_area"]

                    face = img[y1:y2, x1:x2]
                    face = cv2.resize(face, size)
                    face_array = img_to_array(face) / 255.0  
                    face_array = np.array(face_array, dtype=np.float32)
                    data.append((image_path, label, face_array))

    data = pd.DataFrame(data, columns=["image_path", "label", "face_array"])
    
    return data