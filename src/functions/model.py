import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Lambda, BatchNormalization
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K

class FaceModel():
    
    def __init__(self, input_shape = (160, 160, 3), embedding_size = 128):
        """
        Initializes the CNN model for face embedding and triplet network.

        :param input_shape: Shape of input images (height, width, channels)
        :param embedding_size: The output embedding size (default: 128)
        """

        self.input_shape = input_shape
        self.embedding_size = embedding_size
        self.base_model = self.cnn_model()

    def cnn_model(self): 
        """
        Builds the CNN architecture for face embedding extraction.
        """

        model = Sequential([
            Conv2D(32, (3,3), activation = 'relu', input_shape = self.input_shape),
            MaxPooling2D(pool_size = (2,2)),
            
            Conv2D(64, (3,3), activation = 'relu'),
            MaxPooling2D(pool_size = (2,2)),

            Conv2D(128, (3,3), activation = 'relu'),
            MaxPooling2D(pool_size = (2,2)),

            Conv2D(256, (3,3), activation = 'relu'),
            MaxPooling2D(pool_size = (2,2)),

            Flatten(), 
            Dense(512, activation = "relu"),
            Dropout(0.5), 

            Dense(self.embedding_size, activation = "linear", name = "embedding")

        ])

        return model
    
    @staticmethod
    def triplet_loss(y_true, y_pred, alpha = 0.2):
        """
        Computes triplet loss.

        :param y_true: Not used, required for Keras compatibility
        :param y_pred: Concatenated tensor of embeddings (Anchor, Positive, Negative)
        :param alpha: Margin hyperparameter (default=0.2)
        :return: Triplet loss value
        """

        anchor, positive, negative = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:384]
        
        pos_dist = K.sum(K.square(anchor - positive), axis=-1)
        neg_dist = K.sum(K.square(anchor - negative), axis=-1)
        
        loss = K.maximum(pos_dist - neg_dist + alpha, 0)
        return K.mean(loss)

    def triplet_network(self): 
        """
        Creates a triplet network that processes an (Anchor, Positive, Negative) input.
        """
        
        input_anchor = Input(shape = self.input_shape, name = "input_anchor")
        input_pos = Input(shape = self.input_shape, name = "input_pos")
        input_neg = Input(shape = self.input_shape, name = "input_neg")

        anchor_emb = self.base_model(input_anchor)
        pos_emb = self.base_model(input_pos)
        neg_emb = self.base_model(input_neg)

        output = Lambda(lambda x: K.concatenate(x, axis = -1))([anchor_emb, pos_emb, neg_emb])

        return Model(inputs = [input_anchor, input_pos, input_neg], outputs = output, name = "TripletNetwork")