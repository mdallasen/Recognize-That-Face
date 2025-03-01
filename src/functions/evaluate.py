import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf

class Evaluator:
    def __init__(self, model, encoder_model, test_data, test_labels):
        """
        :param model: Trained triplet model
        :param encoder_model: Encoder part of the model for embedding extraction
        :param test_data: Test dataset
        :param test_labels: Corresponding test labels
        """
        self.model = model
        self.encoder_model = encoder_model
        self.test_data = np.array(test_data)
        self.test_labels = np.array(test_labels)

    def compute_triplet_loss(self, anchors, positives, negatives):
        """
        Computes triplet loss for a given set of triplets.
        """
        anchor_embeddings = self.encoder_model.predict(anchors)
        positive_embeddings = self.encoder_model.predict(positives)
        negative_embeddings = self.encoder_model.predict(negatives)

        pos_dist = np.linalg.norm(anchor_embeddings - positive_embeddings, axis=1)
        neg_dist = np.linalg.norm(anchor_embeddings - negative_embeddings, axis=1)
        
        triplet_loss = np.maximum(pos_dist - neg_dist + 0.2, 0)  # 0.2 is margin

        return np.mean(triplet_loss)

    def visualize_embeddings(self, num_samples=500, method="pca"):
        """
        Visualizes the learned embeddings using PCA or t-SNE.
        :param num_samples: Number of samples to plot (default 500).
        :param method: "pca" or "tsne" for visualization.
        """
        indices = np.random.choice(len(self.test_data), num_samples, replace=False)
        selected_data = self.test_data[indices]
        selected_labels = self.test_labels[indices]

        embeddings = self.encoder_model.predict(selected_data)

        if method == "pca":
            reduced_embeddings = PCA(n_components=2).fit_transform(embeddings)
        elif method == "tsne":
            reduced_embeddings = TSNE(n_components=2, perplexity=30).fit_transform(embeddings)
        else:
            raise ValueError("Method should be 'pca' or 'tsne'.")

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=selected_labels, cmap="viridis", alpha=0.7)
        plt.colorbar(scatter, label="Class Labels")
        plt.title(f"Embedding Visualization using {method.upper()}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()

    def plot_distance_distribution(self):
        """
        Plots a histogram of distances between positive and negative pairs.
        """
        embeddings = self.encoder_model.predict(self.test_data)
        distances = pairwise_distances(embeddings, metric="euclidean")

        pos_distances = []
        neg_distances = []

        for i in range(len(self.test_data)):
            same_label_indices = np.where(self.test_labels == self.test_labels[i])[0]
            diff_label_indices = np.where(self.test_labels != self.test_labels[i])[0]

            if len(same_label_indices) > 1:
                pos_distances.append(np.min(distances[i, same_label_indices][1:]))  # Exclude self-distance

            if len(diff_label_indices) > 0:
                neg_distances.append(np.min(distances[i, diff_label_indices]))

        plt.figure(figsize=(8, 6))
        plt.hist(pos_distances, bins=50, alpha=0.7, label="Positive Pairs", color="blue")
        plt.hist(neg_distances, bins=50, alpha=0.7, label="Negative Pairs", color="red")
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Frequency")
        plt.title("Distance Distribution of Positive and Negative Pairs")
        plt.legend()
        plt.show()

    def evaluate(self, test_triplets):
        """
        Evaluates the model with triplet loss and visualization.
        :param test_triplets: Tuple of (anchors, positives, negatives).
        """
        anchors, positives, negatives = test_triplets

        print("\n Evaluating Model Performance...\n")

        # Compute and display triplet loss
        loss = self.compute_triplet_loss(anchors, positives, negatives)
        print(f"🔹 Mean Triplet Loss: {loss:.4f}")

        # Visualize embeddings
        self.visualize_embeddings(method="pca")
        self.visualize_embeddings(method="tsne")

        # Plot distance distributions
        self.plot_distance_distribution()

        print("\n Evaluation Completed.")
