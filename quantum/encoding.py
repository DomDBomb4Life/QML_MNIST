import pennylane as qml
import numpy as np

class QuantumDataEncoder:
    def __init__(self, encoding_type='angle'):
        self.encoding_type = encoding_type.lower()

    def encode(self, x, wires):
        x = np.array(x, dtype=float)
        num_qubits = len(wires)

        if self.encoding_type == 'angle':
            # Check non-empty input
            if x.size == 0:
                raise ValueError("Input vector is empty for angle encoding.")
            # Normalize for angle encoding
            norm_x = np.pi * (x - x.min()) / (x.max() - x.min() + 1e-9)
            for i, val in enumerate(norm_x):
                qml.RX(val, wires=wires[i % num_qubits])

        elif self.encoding_type == 'amplitude':
            # Match size to num_qubits
            if x.size < num_qubits:
                padded = np.zeros(num_qubits)
                padded[:x.size] = x
                x = padded
            elif x.size > num_qubits:
                x = x[:num_qubits]
            vector = x / np.sqrt((x ** 2).sum() + 1e-9)
            # Use features= instead of vector=
            qml.AmplitudeEmbedding(features=vector, wires=wires, normalize=False)

        elif self.encoding_type == 'basis':
            # Ensure binary input
            if not np.all((x == 0) | (x == 1)):
                x = (x > 0.5).astype(int)
            if x.size < num_qubits:
                bin_padded = np.zeros(num_qubits, dtype=int)
                bin_padded[:x.size] = x
                x = bin_padded
            elif x.size > num_qubits:
                x = x[:num_qubits]
            idx_str = ''.join(str(b) for b in x)
            qml.BasisState(np.array(list(map(int, idx_str)), dtype=int), wires=wires)

        else:
            # Default to angle if unknown
            self.encoding_type = 'angle'
            self.encode(x, wires)