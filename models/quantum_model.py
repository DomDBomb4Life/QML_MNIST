import tensorflow as tf
from quantum.quantum_layer import QuantumLayer

def build_quantum_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    # Reduce input features to match the number of qubits
    x = tf.keras.layers.Dense(4, activation='linear', trainable=False)(x)
    x = QuantumLayer(num_qubits=4)(x)
    x = tf.keras.layers.Dense(10)(x)
    outputs = tf.keras.layers.Activation('softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model