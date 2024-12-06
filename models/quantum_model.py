import tensorflow as tf
import pennylane as qml
import numpy as np
from tensorflow.keras import layers
from quantum.encoding import QuantumDataEncoder
from quantum.quantum_circuit import QuantumCircuitBuilder
from utils.config import Config

class QuantumKerasLayer(layers.Layer):
    def __init__(self, num_qubits, circuit_depth, entanglement, encoding_type, noise_level=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.entanglement = entanglement
        self.encoding_type = encoding_type
        self.noise_level = noise_level

        self.encoder = QuantumDataEncoder(encoding_type=self.encoding_type)
        self.circuit_builder = QuantumCircuitBuilder(num_qubits=self.num_qubits,
                                                     circuit_depth=self.circuit_depth,
                                                     entanglement=self.entanglement,
                                                     encoding_type=self.encoding_type,
                                                     noise_level=self.noise_level)
        self.qnode = self.circuit_builder.build_qnode(self.encoder)

        init_shape = (self.circuit_depth, self.num_qubits, 3)
        self.w = self.add_weight(name='quantum_weights',
                                 shape=init_shape,
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        # Convert tensor to numpy for batch execution with qml.batch_execute
        batch_size = tf.shape(inputs)[0]
        inputs_np = tf.numpy_function(lambda x: x, [inputs], tf.float64)
        inputs_np = tf.reshape(inputs_np, [batch_size, -1])

        def run_batch(inputs_array, weights_array):
            inputs_list = inputs_array.tolist()
            # weights_array is constant for the batch
            tapes = []
            for sample in inputs_list:
                self.qnode.construct([np.array(sample, dtype=float), np.array(weights_array, dtype=float)], {})
                tapes.append(self.qnode.qtape)
            res = qml.interfaces.batch_execute(self.qnode.device, tapes, gradient_fn=None)
            return np.array(res, dtype=float)

        outputs = tf.numpy_function(run_batch, [inputs_np, self.w], tf.float64)
        outputs = tf.reshape(outputs, [batch_size, self.num_qubits])
        return tf.cast(outputs, tf.float32)

def build_quantum_model():
    config = Config()
    num_qubits = config.get_quantum_param('num_qubits', 4)
    circuit_depth = config.get_quantum_param('circuit_depth', 1)
    entanglement = config.get_quantum_param('entanglement', 'linear')
    encoding_type = config.get_quantum_param('encoding', 'angle')
    noise_level = config.get_quantum_param('noise_level', 0.0)

    inputs = tf.keras.Input(shape=(28*28,))
    q_layer = QuantumKerasLayer(num_qubits=num_qubits,
                                circuit_depth=circuit_depth,
                                entanglement=entanglement,
                                encoding_type=encoding_type,
                                noise_level=noise_level)(inputs)

    # Use a Dense layer to map the [batch_size, num_qubits] outputs to 10 classes
    outputs = tf.keras.layers.Dense(10, activation='softmax')(q_layer)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model