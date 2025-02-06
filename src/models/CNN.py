import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ( 
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Add, Activation, GlobalAveragePooling2D
)

def residual_block(x, filters, kernel_size, stride = 1): 
    shortcut = x

    x = Conv2D(filters, kernel_size, strides=stride, padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters, kernel_size, strides=1, padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    
    # Residual Connection
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    
    return x

def CNN(input_shape=(160, 160, 3), num_classes=10):
    """Builds a deep CNN model for face recognition"""
    inputs = Input(shape=input_shape)

    # Initial Convolution
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Residual Blocks
    x = residual_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 128)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 256)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Global Pooling + Dense Layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Output Layer
    outputs = Dense(num_classes, activation="softmax")(x)

    # Build Model
    model = Model(inputs, outputs)
    
    return model