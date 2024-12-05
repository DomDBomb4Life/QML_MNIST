import tensorflow as tf

class Trainer:
    def __init__(self, model, data, epochs=10, batch_size=32):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self):
        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.x_test, self.y_test)
        )
        return history

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")