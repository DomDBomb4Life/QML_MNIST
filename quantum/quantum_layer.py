import tensorflow as tf
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, num_qubits=4):
        super(QuantumLayer, self).__init__()
        self.num_qubits = num_qubits
        self.parameters = ParameterVector('theta', length=num_qubits * 2)
        self.backend = Aer.get_backend('statevector_simulator')

    def build(self, input_shape):
        # Initialize trainable parameters for the quantum circuit
        self.theta = self.add_weight(
            name='theta',
            shape=(self.num_qubits * 2,),
            initializer='random_uniform',
            trainable=True
        )

    def call(self, inputs):
        # Prepare quantum circuit for each input sample
        outputs = []
        for input_sample in inputs:
            # For simplicity, only use the first 'num_qubits' features
            input_data = input_sample[:self.num_qubits]
            qc = QuantumCircuit(self.num_qubits)
            # Encode classical data into quantum states
            for i in range(self.num_qubits):
                qc.rx(float(input_data[i]) * np.pi, i)
            # Apply parameterized rotations
            for i in range(self.num_qubits):
                qc.ry(self.theta[i], i)
                qc.rz(self.theta[i + self.num_qubits], i)
            # Apply entangling gates
            qc.cx(0, 1)
            qc.cx(2, 3)
            # Execute the circuit
            job = execute(qc, backend=self.backend, shots=1024)
            result = job.result()
            counts = result.get_counts(qc)
            # Compute expectation value
            expectation = self.compute_expectation(counts)
            outputs.append([expectation])
        return tf.convert_to_tensor(outputs, dtype=tf.float32)

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