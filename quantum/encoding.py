import pennylane as qml
import numpy as np

class QuantumDataEncoder:
    def __init__(self, encoding_type='angle'):
        self.encoding_type = encoding_type.lower()

    def encode(self, x, wires):
        x = np.array(x, dtype=float)
        num_qubits = len(wires)

        if self.encoding_type == 'angle':
            # Check for non-empty input
            if x.size == 0:
                raise ValueError("Input vector is empty for angle encoding.")
            # Normalize input for angle encoding
            norm_x = np.pi * (x - x.min()) / (x.max() - x.min() + 1e-9)
            for i, val in enumerate(norm_x):
                qml.RX(val, wires=wires[i % num_qubits])

        elif self.encoding_type == 'amplitude':
            # Calculate the required state vector length
            total_state = 2 ** num_qubits
            if x.size < total_state:
                # Pad with zeros if input is smaller than required
                padded = np.zeros(total_state)
                padded[:x.size] = x
                x = padded
            elif x.size > total_state:
                # Truncate if input is larger than required
                x = x[:total_state]
            # Normalize the feature vector
            vector = x / np.sqrt((x ** 2).sum() + 1e-9)
            # Use 'features' argument as per PennyLane's API
            qml.AmplitudeEmbedding(features=vector, wires=wires, normalize=False)

        elif self.encoding_type == 'basis':
            # Ensure binary input for basis encoding
            if not np.all((x == 0) | (x == 1)):
                # Binarize inputs using a threshold of 0.5
                x = (x > 0.5).astype(int)
            if x.size < num_qubits:
                # Pad with zeros if input is smaller than the number of qubits
                bin_padded = np.zeros(num_qubits, dtype=int)
                bin_padded[:x.size] = x
                x = bin_padded
            elif x.size > num_qubits:
                # Truncate if input is larger than the number of qubits
                x = x[:num_qubits]
            # Convert binary array to basis state
            idx_str = ''.join(str(b) for b in x)
            qml.BasisState(np.array(list(map(int, idx_str)), dtype=int), wires=wires)

        else:
            # Default to angle encoding if an unknown encoding type is provided
            self.encoding_type = 'angle'
            self.encode(x, wires)