import os
import logging
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from functions.train import train
from functions.detect import detect_label
from functions.preprocess import preprocess_image
from functions.deploy import recognize
from functions.utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def run_pipeline():
    config = load_config("src/config/config.yml") 

    # Face detection
    dataset_path = config["dataset_path"]
    logging.info(f"Detecting faces in dataset at {dataset_path}...")
    detected_faces = detect_label(dataset_path)

    if detected_faces.empty:
        logging.warning("No faces detected in the dataset.")
        return

    os.makedirs("results", exist_ok=True)
    detected_faces_path = "results/detected_faces.csv"
    detected_faces.to_csv(detected_faces_path, index=False)
    logging.info(f"Detected faces saved to {detected_faces_path}")

    # Preprocess the detected images
    logging.info("Preprocessing images...")
    detected_faces["preprocessed_face"] = detected_faces["image_path"].apply(preprocess_image)

    if detected_faces["preprocessed_face"].empty:
        logging.warning("Preprocessing failed or returned None.")
        return

    preprocessed_path = "results/preprocessed_images.npy"
    np.save(preprocessed_path, np.stack(detected_faces["preprocessed_face"].to_numpy()))
    logging.info(f"Preprocessed images saved to {preprocessed_path}")

    # Recognize faces
    logging.info("Recognizing faces...")
    model = load_model(config["model_path"])
    with open(config["encoder_path"], "rb") as f:
        label_encoder = pickle.load(f)
    recognize(detected_faces, model, label_encoder)

    # Train the model
    logging.info("Training the model...")
    train()

    # Deploy the model
    logging.info("Deploying the model...")
    # Add your deployment code here

    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()