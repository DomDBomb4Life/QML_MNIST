import pennylane as qml
import numpy as np

class QuantumDataEncoder:
    def __init__(self, encoding_type='angle'):
        self.encoding_type = encoding_type.lower()

    def encode(self, x, wires):
        # x is a 1D array of features, wires is the list of qubits
        if self.encoding_type == 'angle':
            # Angle Encoding: encode features as rotation angles
            # Normalize x to [0, pi] to ensure full usage of rotation
            norm_x = np.pi * (x - x.min()) / (x.max() - x.min() + 1e-9)
            for i, val in enumerate(norm_x):
                qml.RX(val, wires=wires[i % len(wires)])
        elif self.encoding_type == 'amplitude':
            # Amplitude Encoding:
            # Requires normalized vector. We'll assume x is already normalized
            # If dimension mismatches qubit count, pad with zeros
            vector = x / np.sqrt((x ** 2).sum() + 1e-9)
            qml.AmplitudeEmbedding(vector=vector, wires=wires, normalize=True)
        elif self.encoding_type == 'basis':
            # Basis Encoding:
            # Convert the feature vector into a binary index and prepare a basis state
            # This is simplistic and generally requires binary features
            idx = int(''.join(str(int(val > 0.5)) for val in x), 2) if len(x) > 0 else 0
            qml.BasisState(np.binary_repr(idx, width=len(wires)), wires=wires)
        else:
            # Default to angle encoding if unknown
            self.encoding_type = 'angle'
            self.encode(x, wires)