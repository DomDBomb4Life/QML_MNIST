# File: models/quantum_modelV2.py
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn.torch import TorchLayer

def build_quantum_model(param_values):
    """
    Build quantum model using directly passed parameters in `param_values` dict:
    Keys: 'num_qubits', 'circuit_depth', 'entanglement', 'encoding', 'noise_level'
    """
    num_qubits = param_values["num_qubits"]
    circuit_depth = param_values["circuit_depth"]
    entanglement = param_values["entanglement"]
    # encoding and noise_level can be used if needed for more advanced circuits
    
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        # Assume encoding is angle by default; can extend if encoding logic needed
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
                    qml.CNOT(wires=[i, i+1])
            elif entanglement == "circular":
                for i in range(num_qubits):
                    qml.CNOT(wires=[i, (i+1)%num_qubits])

        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    weight_shapes = {"weights": (circuit_depth, num_qubits, 3)}
    quantum_layer = TorchLayer(qnode, weight_shapes=weight_shapes)

    class QuantumModel(nn.Module):
        def __init__(self):
            super(QuantumModel, self).__init__()
            self.preprocess = nn.Linear(784, num_qubits)
            self.quantum_layer = quantum_layer
            self.classical_layer = nn.Linear(num_qubits, 10)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.preprocess(x)         # (batch_size, num_qubits)
            x = self.quantum_layer(x)      # (batch_size, num_qubits)
            x = self.classical_layer(x)    # (batch_size, 10)
            x = self.softmax(x)            # probabilities
            return x

    return QuantumModel()