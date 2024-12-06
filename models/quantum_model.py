import tensorflow as tf
import pennylane as qml
import numpy as np
from tensorflow.keras import layers
from quantum.encoding import QuantumDataEncoder
from quantum.quantum_circuit import QuantumCircuitBuilder
from utils.config import Config

class QuantumKerasLayer(layers.Layer):
    def __init__(self, num_qubits, circuit_depth, entanglement, encoding_type, **kwargs):
        super().__init__(**kwargs)
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.entanglement = entanglement
        self.encoding_type = encoding_type
        self.encoder = QuantumDataEncoder(encoding_type=self.encoding_type)
        self.circuit_builder = QuantumCircuitBuilder(num_qubits=self.num_qubits,
                                                     circuit_depth=self.circuit_depth,
                                                     entanglement=self.entanglement,
                                                     encoding_type=self.encoding_type)
        self.qnode = self.circuit_builder.build_qnode(self.encoder)

        # Initialize trainable weights
        self.num_weights = self.circuit_depth * self.num_qubits * 3
        # Weights shaped as [circuit_depth, num_qubits, 3]
        init_shape = (self.circuit_depth, self.num_qubits, 3)
        self.w = self.add_weight(name='quantum_weights',
                                 shape=init_shape,
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        # inputs is expected to be of shape [batch_size, feature_dim]
        # We will run the qnode for each sample in the batch
        # Map_fn or vectorization might be needed for efficiency.
        # Here we use tf.vectorized_map for simplicity.
        def circuit_map(x):
            # x is a single sample
            return self.qnode(x, self.w)

        # vectorized_map applies the circuit to each sample in the batch
        outputs = tf.vectorized_map(circuit_map, inputs)
        # outputs shape: [batch_size, num_qubits]
        # For classification, we might reduce over qubits
        # Let's just take mean over qubits to get a single logit
        return tf.reduce_mean(outputs, axis=1, keepdims=True)

def build_quantum_model():
    config = Config()
    num_qubits = config.get_quantum_param('num_qubits', 4)
    circuit_depth = config.get_quantum_param('circuit_depth', 1)
    entanglement = config.get_quantum_param('entanglement', 'linear')
    encoding_type = config.get_quantum_param('encoding', 'angle')

    inputs = tf.keras.Input(shape=(28*28,))  # For MNIST flattened input
    # Optionally add classical preprocessing layers here, e.g. Dense for dimension reduction
    # For simplicity, directly feed to quantum layer:
    q_layer = QuantumKerasLayer(num_qubits=num_qubits,
                                circuit_depth=circuit_depth,
                                entanglement=entanglement,
                                encoding_type=encoding_type)(inputs)

    # Add a classical output layer
    outputs = tf.keras.layers.Dense(10, activation='softmax')(q_layer)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model