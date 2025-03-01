import yaml
import numpy as np
import cv2
import os
import logging
from functions.preprocessor import Preprocessor
from functions.train import Training
from functions.evaluate import Evaluator
from functions.deploy import Deployer
from functions.model import FaceModel

# Load config file
def load_config(config_path="src/config.yml"):
    """ Load configuration from a YAML file. """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Set up logging
def setup_logging(config):
    """ Configures logging based on user settings. """
    log_level = getattr(logging, config.get("log_level", "INFO").upper())
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    if config.get("save_logs", False):
        log_file = config.get("log_file", "logs/pipeline.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

def main():
    """ Main function to run the complete pipeline: Preprocessing, Training, Evaluation, and Deployment. """

    # Load configuration
    config = load_config()
    setup_logging(config)

    logging.info("Starting Face Recognition Pipeline...")

    dataset_path = config["dataset_path"]
    model_path = config["model_path"]
    encoder_path = config["encoder_path"]
    test_image_path = config["test_image"]
    train_test_data_path = config["train_test_data_path"]  #Load from train_test
    deploy_data_path = config["deploy_data_path"]

    ### **Step 1: Preprocessing Data** ###
    logging.info("Loading and processing dataset...")
    preprocessor = Preprocessor(dataset_path=dataset_path)
    preprocessor.load()  # Load and process images
    preprocessor.encode_labels(save_path=encoder_path)  # Encode labels
    preprocessor.save_processed_data(train_path=train_test_data_path, deploy_path=deploy_data_path)
    logging.info(f"Data preprocessing complete! Train-Test data saved at {train_test_data_path} and Deploy data saved at {deploy_data_path}")

    ### **Step 2: Load Processed Data for Training** ###
    logging.info("Loading processed training data...")
    data = np.load(train_test_data_path)  # Load training/testing set
    X_train, X_test, y_train, y_test = train_test_split(data["faces"], data["labels"], test_size=0.2, stratify=data["labels"], random_state=42)
    logging.info(f"Loaded {len(X_train)} training samples and {len(X_test)} testing samples.")

    ### **Step 3: Train Model** ###
    logging.info("Training Face Recognition Model...")
    trainer = Training(X_train, y_train)
    trainer.train(config)  # Train model using correct dataset
    logging.info(f"Model training complete! Model saved at {model_path}")

    ### **Step 4: Evaluate Model** ###
    logging.info("Evaluating Model Performance...")
    
    # Load trained model
    model = FaceModel(input_shape=tuple(config["input_shape"]))
    triplet_model = model.triplet_network()
    triplet_model.load_weights(model_path)
    
    evaluator = Evaluator(triplet_model, model.base_model, X_test, y_test)
    test_triplets = trainer.generate_triplets(X_test, y_test, num_triplets=500)
    evaluator.evaluate(test_triplets)
    logging.info("Model evaluation complete!")

    ### **Step 5: Deploy Model for Inference** ###
    logging.info("Deploying Model for Inference...")
    deployer = Deployer(model_path=model_path, encoder_path=encoder_path, deploy_data_path=deploy_data_path)

    # Load and preprocess a test image
    if not os.path.exists(test_image_path):
        logging.error(f"Test image not found at {test_image_path}")
        return

    test_image = cv2.imread(test_image_path)
    query_image = preprocessor.detect_faces(test_image)
    
    if query_image is None:
        logging.error("No face detected in the test image.")
        return

    # Generate embedding for the test image
    embedding = deployer.generate_embedding(query_image)
    logging.info(f"Generated embedding for sample image: {embedding.shape}")

    logging.info("Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
