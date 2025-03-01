import numpy as np
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from mtcnn import MTCNN  # Face detector
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

class Deployer:
    def __init__(self, model_path="triplet_model.h5", encoder_path="face_encoder.h5", deploy_data_path="results/faces_deploy.npz", results_folder="results/"):
        """
        Loads the trained model, encoder, and deploy dataset.
        :param model_path: Path to the trained triplet model.
        :param encoder_path: Path to the encoder model.
        :param deploy_data_path: Path to stored face embeddings.
        :param results_folder: Folder to save processed images.
        """
        print("Loading the trained model and encoder...")
        self.model = load_model(model_path)
        self.encoder = load_model(encoder_path)
        self.detector = MTCNN()  # Initialize face detector
        self.results_folder = results_folder

        # Ensure results folder exists
        os.makedirs(results_folder, exist_ok=True)

        # Load reference dataset (known faces and labels)
        deploy_data = np.load(deploy_data_path)
        self.reference_images = deploy_data["faces"]
        self.reference_labels = deploy_data["labels"]
        self.reference_embeddings = self.batch_generate_embeddings(self.reference_images)

        print("Model and face dataset successfully loaded.")

    def detect_faces(self, image):
        """
        Detects faces in an image and returns cropped faces with bounding boxes.
        """
        detections = self.detector.detect_faces(image)
        faces = []
        boxes = []

        for det in detections:
            x, y, w, h = det["box"]
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))  # Resize for model
            faces.append(face)
            boxes.append((x, y, w, h))

        return faces, boxes

    def generate_embedding(self, face):
        """
        Generates an embedding for a given face.
        """
        face = face.astype("float32") / 255.0  # Normalize pixel values
        face = np.expand_dims(face, axis=0)
        embedding = self.encoder.predict(face)[0]
        return embedding

    def batch_generate_embeddings(self, images):
        """
        Generates embeddings for multiple images.
        """
        images = np.array(images).astype("float32") / 255.0
        return self.encoder.predict(images)

    def classify_face(self, face):
        """
        Identifies the closest match for a given face.
        """
        face_embedding = self.generate_embedding(face)
        similarities = cosine_similarity([face_embedding], self.reference_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        best_match_label = self.reference_labels[best_match_idx]
        best_match_score = similarities[best_match_idx]

        return best_match_label, best_match_score

    def recognize_faces(self, image):
        """
        Detects faces in an image, classifies them, and draws bounding boxes.
        """
        faces, boxes = self.detect_faces(image)

        for face, (x, y, w, h) in zip(faces, boxes):
            label, score = self.classify_face(face)

            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display name and confidence score
            text = f"{label} ({score:.2f})"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return image

    def show_recognition(self, image_path, save_output=True):
        """
        Reads an image, recognizes faces, saves, and displays the result.
        :param image_path: Path to the image file.
        :param save_output: Whether to save the processed image.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image {image_path}")
            return

        image = self.recognize_faces(image)

        # Save the processed image
        if save_output:
            output_path = os.path.join(self.results_folder, os.path.basename(image_path))
            cv2.imwrite(output_path, image)
            print(f"Processed image saved at: {output_path}")

        # Convert to RGB for correct display in matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title("Face Recognition Results")
        plt.show()
