import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from detect_label import detect_label
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
dataset_path = "src/dataset"
data = detect_label(dataset_path)

# Extract features and labels
X = np.array(data[["face_array", "image_path"]].tolist()).astype("float32")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["label"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)
X_deploy, X_test, y_deploy, y_test = train_test_split(X_test, y_test, test_size = 0.5, stratify = y_test, random_state = 42)

# Save save data for deployment
np.save("X_deploy.npy", X_deploy)
np.save("y_deploy.npy", y_deploy)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),  
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')  
])

# Compile
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ["accuracy"])

# Train model 
model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test))

# Save model
model.save("face_recognition_model.h5")

# Save encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)