import sys
import time
import numpy as np
from utils.config import Config
from quantum.encoding import QuantumDataEncoder
from quantum.quantum_circuit import QuantumCircuitBuilder
import matplotlib.pyplot as plt

def main():
    config = Config()
    encoding_types = [ 'amplitude', 'basis']
    entanglements = ['linear', 'circular']

    # Test all combinations to validate circuit output
    for enc in encoding_types:
        for ent in entanglements:
            num_qubits = config.get_quantum_param('num_qubits', 4)
            circuit_depth = config.get_quantum_param('circuit_depth', 1)
            noise_level = config.get_quantum_param('noise_level', 0.0)

            print("\nQuantum Circuit Testing")
            print("-----------------------")
            print(f"Encoding: {enc}, Num Qubits: {num_qubits}, Depth: {circuit_depth}, Entanglement: {ent}, Noise: {noise_level}")

            encoder = QuantumDataEncoder(encoding_type=enc)
            circuit_builder = QuantumCircuitBuilder(num_qubits=num_qubits,
                                                    circuit_depth=circuit_depth,
                                                    entanglement=ent,
                                                    encoding_type=enc,
                                                    noise_level=noise_level)
            qnode = circuit_builder.build_qnode(encoder)

            # Single random input sample
            sample = np.random.rand(784)
            weight_shape = (circuit_depth, num_qubits, 3)
            weights = np.random.randn(*weight_shape)

            start_time = time.time()
            result = qnode(sample, weights)
            end_time = time.time()

            print("QNode output:", result)
            print(f"Execution Time: {end_time - start_time:.4f} seconds")
            print("Output shape:", np.array(result).shape)

            # Validate output shape
            assert len(result) == num_qubits, "Output shape must match the number of qubits"

            # Convert result to NumPy array for range checking
            result_np = np.array([tensor.numpy() for tensor in result])

            # Check output range for Pauli-Z expectations [-1, 1]
            assert np.all(result_np <= 1.0) and np.all(result_np >= -1.0), "Output values must be within [-1, 1]"

            # Visualize circuit (single input)
            fig = circuit_builder.visualize_circuit(encoder, sample, weights)
            plt.show()
            print("Circuit Diagram displayed.\n")

if __name__ == '__main__':
    main()