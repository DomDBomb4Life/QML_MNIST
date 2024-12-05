import tensorflow as tf
from utils.config import Config

class Trainer:
    def __init__(self, model, data):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.model = model

    def compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self):
        history = self.model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_test, self.y_test),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE
        )
        return history

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        return test_loss, test_acc