# File: models/quantum_model.py
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn.torch import TorchLayer
from utils.config import Config

def build_quantum_model():
    """
    Quantum-classical hybrid model with classical preprocessing.
    """
    config = Config()
    num_qubits = config.get_quantum_param('num_qubits', 4)
    circuit_depth = config.get_quantum_param('circuit_depth', 1)
    entanglement = config.get_quantum_param('entanglement', 'linear')

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        # Angle encoding
        for i in range(num_qubits):
            qml.RX(inputs[:, i] * torch.pi, wires=i)

        for d in range(circuit_depth):
            layer_weights = weights[d]
            for i in range(num_qubits):
                qml.RX(layer_weights[i, 0], wires=i)
                qml.RY(layer_weights[i, 1], wires=i)
                qml.RZ(layer_weights[i, 2], wires=i)
            if entanglement == "linear":
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    weight_shapes = {"weights": (circuit_depth, num_qubits, 3)}
    quantum_layer = TorchLayer(qnode, weight_shapes=weight_shapes)

    class QuantumModel(nn.Module):
        def __init__(self):
            super(QuantumModel, self).__init__()
            self.preprocess = nn.Linear(784, num_qubits)
            self.quantum_layer = quantum_layer
            self.classical_layer = nn.Linear(num_qubits, 10)

        def forward(self, x):
            x = self.preprocess(x)  # (batch_size, num_qubits)
            x = self.quantum_layer(x)  # (batch_size, num_qubits)
            x = self.classical_layer(x)  # (batch_size, 10)
            return x

    return QuantumModel()