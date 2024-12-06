import sys
import time
import tensorflow as tf
import numpy as np
from utils.config import Config
from quantum.encoding import QuantumDataEncoder
from quantum.quantum_circuit import QuantumCircuitBuilder

def main():
    config = Config()
    encoding_type = config.get_quantum_param('encoding', 'angle')
    num_qubits = config.get_quantum_param('num_qubits', 4)
    circuit_depth = config.get_quantum_param('circuit_depth', 1)
    entanglement = config.get_quantum_param('entanglement', 'linear')

    print("Quantum Circuit Testing")
    print("-----------------------")
    print(f"Encoding: {encoding_type}, Num Qubits: {num_qubits}, Depth: {circuit_depth}, Entanglement: {entanglement}")

    encoder = QuantumDataEncoder(encoding_type=encoding_type)
    circuit_builder = QuantumCircuitBuilder(num_qubits=num_qubits,
                                            circuit_depth=circuit_depth,
                                            entanglement=entanglement,
                                            encoding_type=encoding_type)
    qnode = circuit_builder.build_qnode(encoder)

    # Create random input sample to test circuit
    # For demonstration: a random 784-dimensional vector simulating MNIST flattened data
    sample = np.random.rand(784)

    # Random initial weights for testing
    weight_shape = (circuit_depth, num_qubits, 3)
    weights = np.random.randn(*weight_shape)

    # Evaluate the circuit
    start_time = time.time()
    result = qnode(sample, weights)
    end_time = time.time()

    print("QNode output:", result)
    print(f"Execution Time: {end_time - start_time:.4f} seconds")

    # Allow user to tweak parameters interactively (if desired)
    # For now, just print results. The user can modify config.json and rerun this script.

if __name__ == '__main__':
    main()