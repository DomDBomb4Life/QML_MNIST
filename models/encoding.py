import numpy as np

def encode_data(images, method='angle', num_qubits=4):
    # Flatten images
    flat_images = images.reshape(images.shape[0], -1)
    # For MNIST (28x28=784) and num_qubits=4, we can only encode a small subset of features.
    # Let's just take first 'num_qubits' features after a simple reduction.
    # A more sophisticated approach might be PCA or averaging blocks of pixels.
    selected_features = flat_images[:, :num_qubits]

    if method == 'angle':
        # Map pixel values [0,1] to angles [0, pi]
        angles = selected_features * np.pi
        return angles
    else:
        raise ValueError(f"Encoding method '{method}' not implemented.")