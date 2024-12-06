import pennylane as qml
import numpy as np

class QuantumCircuitBuilder:
    def __init__(self, num_qubits=4, circuit_depth=1, entanglement='linear', encoding_type='angle', noise_level=0.0):
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.entanglement = entanglement
        self.encoding_type = encoding_type
        self.noise_level = noise_level
        self.dev = qml.device('default.mixed' if noise_level > 0 else 'default.qubit', wires=self.num_qubits)

    def _entangle_layer(self):
        # 'full' entanglement removed per instructions
        if self.entanglement == 'linear':
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        elif self.entanglement == 'circular':
            for i in range(self.num_qubits):
                qml.CNOT(wires=[i, (i+1) % self.num_qubits])

    def _parameterized_layer(self, params):
        for i in range(self.num_qubits):
            qml.RX(params[i, 0], wires=i)
            qml.RY(params[i, 1], wires=i)
            qml.RZ(params[i, 2], wires=i)

    def _apply_noise(self):
        if self.noise_level > 0:
            for i in range(self.num_qubits):
                qml.DepolarizingChannel(self.noise_level, wires=i)

    def build_qnode(self, encoder, output_shape=1):
        @qml.qnode(self.dev, interface='tf')
        def circuit(inputs, weights):
            encoder.encode(inputs, wires=range(self.num_qubits))
            for d in range(self.circuit_depth):
                self._parameterized_layer(weights[d])
                self._entangle_layer()
                self._apply_noise()

            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qubits)]
        return circuit

    def num_params(self):
        return self.circuit_depth * self.num_qubits * 3

    def visualize_circuit(self, encoder, sample, weights):
        qnode = self.build_qnode(encoder)
        # Use qml.draw_mpl to avoid printing detailed parameters in console
        fig, ax = qml.draw_mpl(qnode)(sample, weights)
        return fig