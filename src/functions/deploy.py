from tensorflow.keras.models import load_model
import pickle
import cv2
import numpy as np
from retinaface import RetinaFace

def recognize(data, model, label_encoder): 
    for index, row in data.iterrows():
        # Extract variables 
        image_path = row["image_path"]
        face_array = np.expand_dims(row["face_array"], axis=0)

        # Predict class
        predict = model.predict(face_array)
        predict_label = label_encoder.inverse_transform([np.argmax(predict)])[0]  

        # Load image to draw bounding box
        img = cv2.imread(image_path)

        # Detect faces again to get bounding box
        faces = RetinaFace.detect_faces(img) if img is not None else None 
        
        if faces is None or len(faces) == 0: 
            continue

        for _, face_info in faces.items():
            x1, y1, x2, y2 = face_info["facial_area"]

            # Draw bounding box and label 
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, predict_label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Show Image
        cv2.imshow("Face Recognition", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()