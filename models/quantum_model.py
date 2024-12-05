import tensorflow as tf
from quantum.quantum_layer import QuantumLayer

def build_quantum_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        QuantumLayer(num_qubits=4),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model