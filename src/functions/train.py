import numpy as np
from sklearn.model_selection import train_test_split
from functions.model import FaceModel

class Training: 
    def __init__(self, data, labels): 
        self.data = data
        self.labels = labels

    def generate_triplets(self, data, labels, num_triplets = 200):
        """
        Generates (Anchor, Positive, Negative) triplets for training.

        :param num_triplets: Number of triplets to generate (default: 1000).
        :return: Three numpy arrays (anchors, positives, negatives).
        """
        
        unique_labels = np.unique(labels)
        label_dict = {label: np.where(labels == label)[0] for label in unique_labels}

        total_images = sum(len(indices) for indices in label_dict.values())

        if num_triplets > total_images:
            raise ValueError(f"Not enough data to generate {num_triplets} triplets.")
        
        anchors, positives, negatives = [], [], []

        for _ in range(num_triplets): 

            anchor_label = np.random.choice(unique_labels)
            anchor_idx = np.random.choice(label_dict[anchor_label])
            possible_positives = np.setdiff1d(label_dict[anchor_label], anchor_idx)

            if len(possible_positives) == 0:
                continue 
            
            positive_idx = np.random.choice(possible_positives)
            negative_label = np.random.choice(unique_labels[unique_labels != anchor_label])
            negative_idx = np.random.choice(label_dict[negative_label])

            anchors.append(data[anchor_idx])
            positives.append(data[positive_idx])
            negatives.append(data[negative_idx])
    
        return np.array(anchors), np.array(positives), np.array(negatives) 

    def train(self, config):

        model_path = config.get("model_path", "src/models/face_recognition_model.h5")

        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, 
            test_size = 0.2, random_state = 42)

        anchors_train, positives_train, negatives_train = self.generate_triplets(X_train, y_train)
        anchors_test, positives_test, negatives_test = self.generate_triplets(X_test, y_test)

        model = FaceModel()
        triplet_model = model.triplet_network()

        triplet_model.compile(optimizer='adam', loss = model.triplet_loss)

        triplet_model.fit(
            [anchors_train, positives_train, negatives_train],
            np.zeros((anchors_train.shape[0], 1)), 
            epochs=10, 
            batch_size=32, 
            validation_data=(
                [anchors_test, positives_test, negatives_test], 
                np.zeros((anchors_test.shape[0], 1))
            )
        )

        triplet_model.save(model_path)
        print(f"Model saved successfully at: {model_path}")