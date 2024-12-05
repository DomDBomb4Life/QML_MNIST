import tensorflow as tf
import tensorflow_quantum as tfq
from utils.config import Config
from quantum.quantum_circuits import build_parametric_circuit, observable_for_classification
import cirq

def build_quantum_model():
    # Build the parameterized quantum circuit
    circuit, params = build_parametric_circuit(Config.NUM_QUBITS, Config.CIRCUIT_DEPTH)
    obs = observable_for_classification(Config.NUM_QUBITS)

    # Inputs are encoded angles, shape = (batch, num_qubits)
    inputs = tf.keras.Input(shape=(Config.NUM_QUBITS,), dtype=tf.float32)
    
    # Convert classical angles to quantum circuits
    # We'll use angle encoding: Apply rotations on each qubit according to the inputs
    # For example, encode angle into Ry rotation: Ry(input_angle)
    # We can prepend these input-dependent operations to the parameterized circuit
    qubits = [cirq.GridQubit(0, i) for i in range(Config.NUM_QUBITS)]
    input_symbols = [cirq.Symbol(f'input_{i}') for i in range(Config.NUM_QUBITS)]
    encoding_circuit = cirq.Circuit()
    for i, q in enumerate(qubits):
        # Encode the input angle into a rotation about the Y-axis
        encoding_circuit.append(cirq.ry(input_symbols[i])(q))

    # PQC Layer: combines encoding + parameterized circuit
    full_circuit = encoding_circuit + circuit

    # PQC layer from TFQ
    pqc_layer = tfq.layers.PQC(full_circuit, obs)
    quantum_output = pqc_layer(inputs, symbol_names=[str(p) for p in params] + [str(i) for i in input_symbols])

    # quantum_output will be expectation values, shape (batch, 1)
    # Add a Dense layer to classify into NUM_CLASSES
    dense = tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')(quantum_output)

    model = tf.keras.Model(inputs=inputs, outputs=dense)
    return model