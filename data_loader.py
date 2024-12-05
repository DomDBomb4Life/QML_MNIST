import tensorflow as tf
import numpy as np

class DataLoader:
    def __init__(self):
        self.num_classes = 10

    def load_data(self):
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Normalize and flatten images
        x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
        # One-hot encode labels
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        return (x_train, y_train), (x_test, y_test)