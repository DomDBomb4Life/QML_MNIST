import cirq
import numpy as np
from quantum.quantum_circuits import create_qubits, build_parametric_circuit, observable_for_classification
import tensorflow_quantum as tfq

def test_quantum_circuit():
    # Test basic functionality: build a circuit and evaluate on a simple input state.
    num_qubits = 4
    depth = 2
    circuit, params = build_parametric_circuit(num_qubits, depth)

    # Set all parameters to zero just as a test
    resolver = {p: 0.0 for p in params}

    # Create a simple input state: |0...0>
    input_circuit = cirq.Circuit()  # no additional gates, means |0...0> state

    # Create a tfq input
    pqc = tfq.convert_to_tensor([circuit])
    inputs = tfq.convert_to_tensor([input_circuit])

    # Compute expectation of the observable
    obs = observable_for_classification(num_qubits)
    expectation_layer = tfq.layers.Expectation()
    exp_values = expectation_layer(
        inputs=inputs, symbol_names=[str(p) for p in params], symbol_values=[[0.0]*len(params)], operators=obs
    )

    print("Test Quantum Circuit - Expectation Value at Zero Params:", exp_values.numpy())

if __name__ == '__main__':
    test_quantum_circuit()