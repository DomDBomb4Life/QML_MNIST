import os
import json
import numpy as np
import tensorflow as tf
import pennylane as qml

class Trainer:
    def __init__(self, model, data, data_loader, mode='classical', epochs=10, batch_size=32, optimizer='adam', learning_rate=0.001, results_dir='results'):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.model = model
        self.data_loader = data_loader
        self.mode = mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate
        self.results_dir = results_dir
        self.logs_path = os.path.join(self.results_dir, 'logs', f'{self.mode}_training_logs.json')
        self.metrics_history = {"epoch": [], "train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    def compile_model(self):
        if self.mode == 'classical':
            opt = self._get_optimizer(self.optimizer_type, self.learning_rate)
            self.model.compile(
                optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

    def train(self):
        if self.mode == 'classical':
            datagen = self.data_loader.get_data_generator()
            datagen.fit(self.x_train)
            history = self.model.fit(
                datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                steps_per_epoch=len(self.x_train) // self.batch_size,
                epochs=self.epochs,
                validation_data=(self.x_test, self.y_test),
                verbose=0
            )
            self._record_history_keras(history)
            return history
        else:
            return self._train_quantum()

    def _train_quantum(self):
        q_params = [v for v in self.model.trainable_variables if 'quantum_weights' in v.name]
        c_params = [v for v in self.model.trainable_variables if 'quantum_weights' not in v.name]

        q_optimizer = qml.Adam(stepsize=self.learning_rate)
        c_optimizer = tf.keras.optimizers.get({'class_name': self.optimizer_type, 'config': {'learning_rate': self.learning_rate}})

        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(1024).batch(self.batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(self.batch_size)

        def forward_loss(x, y, q_params_flat):
            self._set_quantum_params(q_params, q_params_flat)
            preds = self.model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, preds))
            acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y, preds))
            return loss, acc

        q_params_flat = self._flatten_params(q_params)
        for epoch in range(self.epochs):
            epoch_loss, epoch_acc, count = 0, 0, 0
            for x_batch, y_batch in train_ds:
                with tf.GradientTape() as tape:
                    self._set_quantum_params(q_params, q_params_flat)
                    preds = self.model(x_batch, training=True)
                    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, preds))
                grads = tape.gradient(loss, c_params + q_params)
                c_grads = grads[:len(c_params)]
                q_grads = grads[len(c_params):]

                c_optimizer.apply_gradients(zip(c_grads, c_params))
                q_params_flat = q_optimizer.step(lambda p: self._quantum_cost(p, x_batch, y_batch, q_params), q_params_flat)

                acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_batch, preds))
                epoch_loss += loss.numpy() * len(x_batch)
                epoch_acc += acc.numpy() * len(x_batch)
                count += len(x_batch)

            train_loss = epoch_loss / count
            train_acc = epoch_acc / count

            val_loss, val_acc = self._evaluate_dataset(val_ds, q_params, q_params_flat)
            self._log_epoch(epoch+1, train_loss, train_acc, val_loss, val_acc)
        self._save_logs()
        return None

    def _quantum_cost(self, q_params_flat, x, y, q_params):
        self._set_quantum_params(q_params, q_params_flat)
        preds = self.model(x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, preds))
        return loss.numpy()

    def _set_quantum_params(self, q_params, q_params_flat):
        idx = 0
        for v in q_params:
            shape = v.shape
            size = np.prod(shape)
            v.assign(tf.reshape(tf.constant(q_params_flat[idx:idx+size], dtype=tf.float32), shape))
            idx += size

    def _flatten_params(self, q_params):
        return np.concatenate([p.numpy().flatten() for p in q_params])

    def _evaluate_dataset(self, ds, q_params, q_params_flat):
        self._set_quantum_params(q_params, q_params_flat)
        total_loss, total_acc, count = 0, 0, 0
        for x_batch, y_batch in ds:
            preds = self.model(x_batch, training=False)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, preds)).numpy()
            acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_batch, preds)).numpy()
            total_loss += loss * len(x_batch)
            total_acc += acc * len(x_batch)
            count += len(x_batch)
        return total_loss/count, total_acc/count

    def _log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_accuracy'].append(train_acc)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['val_accuracy'].append(val_acc)

    def _record_history_keras(self, history):
        for i, epoch_num in enumerate(history.epoch):
            self.metrics_history['epoch'].append(epoch_num+1)
            self.metrics_history['train_loss'].append(history.history['loss'][i])
            self.metrics_history['train_accuracy'].append(history.history['accuracy'][i])
            self.metrics_history['val_loss'].append(history.history['val_loss'][i])
            self.metrics_history['val_accuracy'].append(history.history['val_accuracy'][i])
        self._save_logs()

    def _save_logs(self):
        with open(self.logs_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)

    def evaluate(self):
        if self.mode == 'classical':
            results = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            # Already logged metrics in train
        else:
            # Quantum evaluation handled externally
            pass

    def _get_optimizer(self, optimizer_type, lr):
        if optimizer_type.lower() == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_type.lower() == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=lr)
        else:
            return tf.keras.optimizers.Adam(learning_rate=lr)