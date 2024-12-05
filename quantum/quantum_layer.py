import tensorflow as tf
import numpy as np
from qiskit import QuantumCircuit, Aer, execute

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, num_qubits=4):
        super(QuantumLayer, self).__init__()
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('statevector_simulator')

    def build(self, input_shape):
        # Initialize trainable parameters for the quantum circuit
        self.theta = self.add_weight(
            name='theta',
            shape=(self.num_qubits * 2,),
            initializer='random_uniform',
            trainable=False  # Set to False since we'll optimize externally
        )

    def call(self, inputs):
        outputs = tf.py_function(
            func=self.quantum_computation,
            inp=[inputs, self.theta],
            Tout=tf.float32
        )
        outputs.set_shape((inputs.shape[0], 1))
        # Stop gradient flow after this layer
        outputs = tf.stop_gradient(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def quantum_computation(self, inputs, theta):
        # Convert inputs and parameters to numpy arrays
        inputs = inputs.numpy()
        theta = theta.numpy()
        batch_size = inputs.shape[0]
        outputs = []
        for idx in range(batch_size):
            input_sample = inputs[idx]
            # Use the first 'num_qubits' features
            input_data = input_sample[:self.num_qubits]
            qc = QuantumCircuit(self.num_qubits)
            # Encode classical data into quantum states
            for i in range(self.num_qubits):
                qc.rx(float(input_data[i]) * np.pi, i)
            # Apply parameterized rotations
            for i in range(self.num_qubits):
                qc.ry(float(theta[i]), i)
                qc.rz(float(theta[i + self.num_qubits]), i)
            # Add entangling gates
            qc.cx(0, 1)
            qc.cx(2, 3)
            # Execute the circuit
            job = execute(qc, backend=self.backend, shots=1024)
            result = job.result()
            counts = result.get_counts(qc)
            # Compute expectation value
            expectation = self.compute_expectation(counts)
            outputs.append([expectation])
        return np.array(outputs, dtype=np.float32)

    def compute_expectation(self, counts):
        # Compute expectation value of Z^{\otimes n}
        expectation = 0
        total_counts = sum(counts.values())
        for outcome, count in counts.items():
            bitstring = outcome[::-1]  # Reverse to match qubit order
            z_measurement = 1
            for bit in bitstring:
                if bit == '1':
                    z_measurement *= -1
                else:
                    z_measurement *= 1
            expectation += z_measurement * (count / total_counts)
        return expectation