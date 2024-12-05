import tensorflow as tf
import numpy as np

class DataLoader:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Shape adjustment for potential CNN layers or future modifications
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        return (x_train, y_train), (x_test, y_test)

    def get_data_generator(self):
        # Data augmentation for better generalization
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        return datagen