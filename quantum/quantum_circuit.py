import pennylane as qml
import numpy as np

class QuantumCircuitBuilder:
    def __init__(self, num_qubits=4, circuit_depth=1, entanglement='linear', encoding_type='angle'):
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.entanglement = entanglement
        self.encoding_type = encoding_type
        self.dev = qml.device('default.qubit', wires=self.num_qubits)

    def _entangle_layer(self):
        # Apply entangling gates according to the chosen entanglement pattern
        if self.entanglement == 'linear':
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        elif self.entanglement == 'circular':
            for i in range(self.num_qubits):
                qml.CNOT(wires=[i, (i+1) % self.num_qubits])
        elif self.entanglement == 'full':
            # Fully connected entanglement (not efficient for large qubit counts)
            for i in range(self.num_qubits):
                for j in range(i+1, self.num_qubits):
                    qml.CNOT(wires=[i, j])

    def _parameterized_layer(self, params):
        # params expected to be an array of shape [self.num_qubits, 3] for RX, RY, RZ
        for i in range(self.num_qubits):
            qml.RX(params[i, 0], wires=i)
            qml.RY(params[i, 1], wires=i)
            qml.RZ(params[i, 2], wires=i)

    def build_qnode(self, encoder, output_shape=1):
        @qml.qnode(self.dev, interface='tf')
        def circuit(inputs, weights):
            # inputs: single sample input
            # weights: trainable parameters for PQC
            encoder.encode(inputs, wires=range(self.num_qubits))

            # Apply multiple layers of parameterized rotations and entanglements
            depth = self.circuit_depth
            for d in range(depth):
                self._parameterized_layer(weights[d])
                self._entangle_layer()

            # Measure expectation values of Pauli-Z on all qubits and combine results
            # For classification, often measuring one qubit suffices, but we can measure multiple
            # and return a single expectation (e.g., mean)
            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qubits)]
        return circuit

    def num_params(self):
        # For each layer, we have self.num_qubits * 3 parameters (RX, RY, RZ)
        # Total layers = circuit_depth
        return self.circuit_depth * self.num_qubits * 3