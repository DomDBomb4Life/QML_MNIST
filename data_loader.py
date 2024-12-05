import tensorflow as tf
import numpy as np
from utils.config import Config
from models.encoding import encode_data

class DataLoader:
    def __init__(self):
        self.num_classes = Config.NUM_CLASSES

    def load_data(self):
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # One-hot encode labels
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)

        # Encode data into a suitable quantum representation
        # Here, we will simply flatten and encode angle-wise.
        x_train_encoded = encode_data(x_train, method=Config.ENCODING_METHOD, num_qubits=Config.NUM_QUBITS)
        x_test_encoded = encode_data(x_test, method=Config.ENCODING_METHOD, num_qubits=Config.NUM_QUBITS)

        return (x_train_encoded, y_train), (x_test_encoded, y_test)