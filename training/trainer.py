import tensorflow as tf
import numpy as np
from quantum.quantum_optimizer import QuantumOptimizer
from quantum.quantum_layer import QuantumLayer

class Trainer:
    def __init__(self, model, data, data_loader, epochs=10, batch_size=32, mode='classical'):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.data_loader = data_loader
        self.mode = mode

        if self.mode == 'quantum':
            # Find the QuantumLayer in the model
            self.quantum_layer = None
            for layer in self.model.layers:
                if isinstance(layer, QuantumLayer):
                    self.quantum_layer = layer
                    break
            if self.quantum_layer is None:
                raise ValueError('QuantumLayer not found in the model.')

            self.theta = self.quantum_layer.theta.numpy()
            self.quantum_optimizer = QuantumOptimizer(num_qubits=self.quantum_layer.num_qubits)
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
            self.metric = tf.keras.metrics.CategoricalAccuracy()
            self.optimizer = tf.keras.optimizers.Adam()
        else:
            self.optimizer = None  # Not used in classical mode

    def compile_model(self):
        if self.mode == 'classical':
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
        else:
            # For quantum mode, we'll manage optimization manually
            pass

    def train(self):
        if self.mode == 'classical':
            datagen = self.data_loader.get_data_generator()
            datagen.fit(self.x_train)
            history = self.model.fit(
                datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                steps_per_epoch=len(self.x_train) // self.batch_size,
                epochs=self.epochs,
                validation_data=(self.x_test, self.y_test)
            )
            return history
        else:
            # Custom training loop for quantum model
            history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
            datagen = self.data_loader.get_data_generator()
            datagen.fit(self.x_train)
            for epoch in range(self.epochs):
                print(f"Epoch {epoch+1}/{self.epochs}")
                for step, (x_batch, y_batch) in enumerate(
                    datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size)
                ):
                    # Forward pass
                    with tf.GradientTape() as tape:
                        predictions = self.model(x_batch, training=True)
                        loss = self.loss_fn(y_batch, predictions)
                    # Update non-quantum weights
                    trainable_vars = [
                        var for var in self.model.trainable_variables
                        if var.trainable and var is not self.quantum_layer.theta
                    ]
                    gradients = tape.gradient(loss, trainable_vars)
                    # Filter out None gradients
                    gradients_vars = [
                        (grad, var) for grad, var in zip(gradients, trainable_vars) if grad is not None
                    ]
                    self.optimizer.apply_gradients(gradients_vars)
                    # Quantum optimization
                    cost_value = loss.numpy()
                    optimal_theta = self.quantum_optimizer.optimize(cost_value, self.theta)
                    self.quantum_layer.theta.assign(optimal_theta)
                    self.theta = optimal_theta
                    # Update metrics
                    self.metric.update_state(y_batch, predictions)
                # Record metrics
                train_acc = self.metric.result().numpy()
                self.metric.reset_states()
                history['accuracy'].append(train_acc)
                history['loss'].append(loss.numpy())
                # Validation
                val_predictions = self.model(self.x_test, training=False)
                val_loss = self.loss_fn(self.y_test, val_predictions).numpy()
                val_acc = tf.keras.metrics.categorical_accuracy(self.y_test, val_predictions)
                val_acc = tf.reduce_mean(val_acc).numpy()
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                print(f"Loss: {loss.numpy():.4f}, Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
            return history

    def evaluate(self):
        if self.mode == 'classical':
            test_loss, test_acc, test_precision, test_recall = self.model.evaluate(self.x_test, self.y_test)
            print(f"Test Accuracy: {test_acc * 100:.2f}%")
            print(f"Test Precision: {test_precision * 100:.2f}%")
            print(f"Test Recall: {test_recall * 100:.2f}%")
        else:
            # Evaluate quantum model
            predictions = self.model(self.x_test, training=False)
            test_loss = self.loss_fn(self.y_test, predictions).numpy()
            test_accuracy = tf.keras.metrics.categorical_accuracy(self.y_test, predictions)
            test_accuracy = tf.reduce_mean(test_accuracy).numpy()
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy * 100:.2f}%")