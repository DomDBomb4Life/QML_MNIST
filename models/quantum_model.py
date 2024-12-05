import tensorflow as tf
from tensorflow.keras.layers import Layer # type: ignore
from quantum.quantum_layer import QuantumLayer

def build_quantum_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        QuantumLayer(num_qubits=4, backend='aer_simulator_statevector'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model