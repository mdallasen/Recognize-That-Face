import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tensorflow.keras.models import load_model

class Deployer:
    def __init__(self, model_path="triplet_model.h5", encoder_path="face_encoder.h5"):
        """
        Loads the trained model and encoder for inference.
        :param model_path: Path to the trained triplet model.
        :param encoder_path: Path to the encoder model.
        """
        print("Loading the trained model and encoder...")
        self.model = load_model(model_path)
        self.encoder = load_model(encoder_path)
        print("Model successfully loaded.")

    def generate_embedding(self, image):
        """
        Generates an embedding for a given image.
        :param image: Input image (preprocessed).
        :return: Embedding vector.
        """
        image = np.expand_dims(image, axis=0) 
        embedding = self.encoder.predict(image)[0]  
        return embedding

    def verify_faces(self, image1, image2, threshold=0.5, metric="cosine"):
        """
        Compares two face images and determines if they belong to the same person.
        :param image1: First image (preprocessed).
        :param image2: Second image (preprocessed).
        :param threshold: Similarity threshold for verification.
        :param metric: Distance metric ("cosine" or "euclidean").
        :return: Boolean indicating whether the faces match.
        """
        emb1 = self.generate_embedding(image1)
        emb2 = self.generate_embedding(image2)

        if metric == "cosine":
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return similarity > threshold, similarity
        elif metric == "euclidean":
            distance = euclidean_distances([emb1], [emb2])[0][0]
            return distance < threshold, distance
        else:
            raise ValueError("Invalid metric. Choose 'cosine' or 'euclidean'.")

    def batch_generate_embeddings(self, images):
        """
        Generates embeddings for a batch of images.
        :param images: List or batch of preprocessed images.
        :return: Numpy array of embeddings.
        """
        return self.encoder.predict(np.array(images))
    
    def find_nearest_neighbors(self, query_image, reference_images, reference_labels, top_k=5):
        """
        Finds the closest matching images from a reference set.
        """
        query_embedding = self.generate_embedding(query_image)
        ref_embeddings = self.batch_generate_embeddings(reference_images)

        distances = np.linalg.norm(ref_embeddings - query_embedding, axis=1)
        closest_indices = np.argsort(distances)[:top_k]
        closest_labels = [reference_labels[i] for i in closest_indices]
        
        return closest_labels 

    def visualize_matches(self, query_image, reference_images, reference_labels, top_k=5):
        """
        Displays the query image alongside its closest matches.
        :param query_image: The input image.
        :param reference_images: Dataset of reference images.
        :param reference_labels: Corresponding labels.
        :param top_k: Number of matches to display.
        """
        closest_indices = self.find_nearest_neighbors(query_image, reference_images, reference_labels, top_k)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, top_k + 1, 1)
        plt.imshow(query_image.squeeze(), cmap="gray")
        plt.title("Query Image")
        plt.axis("off")

        for i, idx in enumerate(closest_indices):
            plt.subplot(1, top_k + 1, i + 2)
            plt.imshow(reference_images[idx].squeeze(), cmap="gray")
            plt.title(f"Match {i+1}\nLabel: {reference_labels[idx]}")
            plt.axis("off")

        plt.show()