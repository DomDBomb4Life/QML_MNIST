import tensorflow as tf
from quantum.quantum_optimizer import QuantumOptimizer

class Trainer:
    def __init__(self, model, data, data_loader, epochs=10, batch_size=32, mode='classical'):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.data_loader = data_loader
        self.mode = mode
        if self.mode == 'quantum':
            self.quantum_optimizer = QuantumOptimizer(self.model.layers[3])  # Assuming QuantumLayer is at index 3

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

    def train(self):
        datagen = self.data_loader.get_data_generator()
        datagen.fit(self.x_train)
        history = None
        if self.mode == 'classical':
            history = self.model.fit(
                datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                steps_per_epoch=len(self.x_train) // self.batch_size,
                epochs=self.epochs,
                validation_data=(self.x_test, self.y_test)
            )
        elif self.mode == 'quantum':
            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}/{self.epochs}")
                batches = datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size)
                for step in range(len(self.x_train) // self.batch_size):
                    x_batch, y_batch = next(batches)
                    with tf.GradientTape() as tape:
                        predictions = self.model(x_batch, training=True)
                        loss = self.model.compiled_loss(y_batch, predictions)
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    # Apply classical gradients
                    self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    # Perform quantum optimization on QuantumLayer parameters
                    self.quantum_optimizer.optimize(loss_function=loss)
                # Validation step
                val_loss, val_acc, val_precision, val_recall = self.model.evaluate(self.x_test, self.y_test, verbose=0)
                print(f"Validation Accuracy: {val_acc * 100:.2f}%")
        return history

    def evaluate(self):
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(self.x_test, self.y_test)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        print(f"Test Precision: {test_precision * 100:.2f}%")
        print(f"Test Recall: {test_recall * 100:.2f}%")