import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from functions.detect import detect_label 
from functions.utils import save_model, load_config
from models.CNN import CNN

# Load dataset
def load_dataset(dataset_path):
    data = detect_label(dataset_path)

    # Extract features and labels
    X = np.array(data["face_array"].tolist()).astype("float32")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["label"])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_deploy, X_test, y_deploy, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=42)

    # Save data for deployment
    np.save("models/X_deploy.npy", X_deploy)
    np.save("models/y_deploy.npy", y_deploy)

    return X_train, X_test, y_train, y_test, label_encoder

def train(config):
    dataset_path = config["dataset_path"]
    model_path = config["model_path"]
    encoder_path = config["encoder_path"]

    X_train, X_test, y_train, y_test, label_encoder = load_dataset(dataset_path)

    model = CNN(input_shape=(160, 160, 3), num_classes=len(label_encoder.classes_))

    # Compile
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    # Train model 
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save model
    model.save(model_path)

    # Save encoder
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)