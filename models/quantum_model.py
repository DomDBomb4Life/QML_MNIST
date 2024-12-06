# File: models/quantum_model.py
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn.torch import TorchLayer
from utils.config import Config

def build_quantum_model():
    """
    Builds a quantum-classical hybrid model using PennyLane's TorchLayer integrated with PyTorch.
    Inputs should be (batch_size, 784) after flattening.
    Output is (batch_size, 10) logits for classification.
    """
    config = Config()
    num_qubits = config.get_quantum_param('num_qubits', 4)
    circuit_depth = config.get_quantum_param('circuit_depth', 1)
    entanglement = config.get_quantum_param('entanglement', 'linear')
    # encoding_type and noise_level are unused directly in this code snippet, but retained for completeness.
    encoding_type = config.get_quantum_param('encoding', 'angle')
    noise_level = config.get_quantum_param('noise_level', 0.0)

    dev = qml.device('default.qubit', wires=num_qubits)

    @qml.qnode(dev, interface='torch')
    def qnode(inputs, weights):
        # Angle encoding for first 'num_qubits' features
        for i in range(num_qubits):
            qml.RX(inputs[i] * torch.pi, wires=i)

        # Parameterized layers
        for d in range(circuit_depth):
            layer_weights = weights[d]
            for i in range(num_qubits):
                qml.RX(layer_weights[i, 0], wires=i)
                qml.RY(layer_weights[i, 1], wires=i)
                qml.RZ(layer_weights[i, 2], wires=i)
            if entanglement == 'linear':
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
            elif entanglement == 'circular':
                for i in range(num_qubits):
                    qml.CNOT(wires=[i, (i+1) % num_qubits])

        # Return expectation values for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    weight_shapes = {"weights": (circuit_depth, num_qubits, 3)}
    quantum_layer = TorchLayer(qnode, weight_shapes=weight_shapes)

    class QuantumModel(nn.Module):
        def __init__(self):
            super(QuantumModel, self).__init__()
            self.quantum_layer = quantum_layer
            self.classical_layer = nn.Linear(num_qubits, 10)

        def forward(self, x):
            # x shape: (batch_size, 784)
            x = self.quantum_layer(x)        # (batch_size, num_qubits)
            x = self.classical_layer(x)      # (batch_size, 10)
            # No softmax here, CrossEntropyLoss expects logits
            return x

    return QuantumModel()