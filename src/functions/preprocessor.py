import os
import cv2
import numpy as np
from mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
import pickle

class Preprocessor: 
    def __init__(self, dataset_path = None, model_path = None, econder_path = None): 
        """ 
        Initializes the Preprocessor class 

        :param dataset_path: Path to dataset folder
        :param model_path: Path to trained model
        :param encoder_path: Path to label encoder file
        """
        self.dataset_path = dataset_path
        self.data = []
        self.labels = []
        self.face_detector = MTCNN()

    def load(self): 
        """
        Loads images and labels from the dataset folder, detects faces, and stores processed data.
        """

        if not self.dataset_path: 
            raise ValueError("Dataset path not found")
        
        for label in os.listdir(self.dataset_path): 
            
            label_path = os.path.join(self.dataset_path, label) 

            if os.path.isdir(label_path): 

                for image_file in os.listdir(label_path): 
                    
                    image_path = os.path.join(label_path, image_file)
                    img = cv2.imread(image_path)

                    if img is None: 
                        print(f"Warning: Unable to read {image_path}")
                        continue 
                
                    face = self.face_detector.detect_faces(img)

                    if face is not None: 
                        self.data.append(face)
                        self.labels.append(label)
            
        print(f"Loaded {len(self.data)} images with detected faces.")


    def detect_faces(self, img, target_size = (160, 160)): 
        """
        Detects a face in an image, crops it, and resizes it.

        :param img: Input image (BGR format from OpenCV)
        :param target_size: Desired output size of the cropped face
        :return: Cropped and resized face or None if no face is detected
        """

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detect_faces(img_rgb)

        if len(faces) > 0: 
            x, y, w, h = faces[0]["box"]
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, target_size)
            return face 
        else: 
            print("No face detected")
            return None
    
    def encode_labels(self, save_path = "label_encoder.pkl"): 
        """
        Encodes string labels into numeric values and saves the encoder.

        :param save_path: Path to save the label encoder
        """

        if not self.labels: 
            raise ValueError("No labels found")

        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        with open(save_path, "wb") as f: 
            pickle.dump(self.label_encoder, f)

        print(f"Label encoder saved to {save_path}.")

    def save_processed_data(self, save_path = "processed_faces.npy"): 

        if not self.data: 
            raise ValueError("No face data found")

        np.savez(save_path, faces = np.array(self.data), labels = np.array(self.encoded_labels))

        print(f"Processed data saved to {save_path}")
    