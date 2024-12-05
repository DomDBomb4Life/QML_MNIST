import cirq
import sympy
import tensorflow_quantum as tfq
from utils.config import Config

def create_qubits(num_qubits):
    return [cirq.GridQubit(0, i) for i in range(num_qubits)]

def build_parametric_circuit(num_qubits, depth):
    qubits = create_qubits(num_qubits)
    circuit = cirq.Circuit()
    # Parameter symbols for rotation gates
    params = sympy.symbols('theta0:'+str(num_qubits*depth*3))

    # We'll apply a pattern of RX, RY, RZ to each qubit per layer, plus entangling CNOTs
    param_index = 0
    for d in range(depth):
        # Single-qubit rotations
        for q in range(num_qubits):
            circuit.append(cirq.rx(params[param_index])(qubits[q]))
            param_index += 1
            circuit.append(cirq.ry(params[param_index])(qubits[q]))
            param_index += 1
            circuit.append(cirq.rz(params[param_index])(qubits[q]))
            param_index += 1
        # Add entangling layer (CNOT chain)
        for q in range(num_qubits - 1):
            circuit.append(cirq.CNOT(qubits[q], qubits[q+1]))

    return circuit, list(params)

def observable_for_classification(num_qubits):
    # For simplicity, measure in Z-basis all qubits.
    # We'll take expectation values and use them as features for classification.
    return sum([cirq.Z(q) for q in create_qubits(num_qubits)])