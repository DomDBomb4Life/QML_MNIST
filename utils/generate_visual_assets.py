import os
import json
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from textwrap import dedent

# Ensure output directory
os.makedirs("assets", exist_ok=True)

def draw_classical_model_diagram():
    """
    Draws and saves a diagram of the ClassicalModel architecture.
    Layers: Flatten -> Dense(784->64) -> ReLU -> Dense(64->10) -> Softmax
    """
    fig, ax = plt.subplots(figsize=(10, 3))

    # Define layer positions
    layers = [
        ("Flatten", "lightblue", (0.1, 0.4, 0.2, 0.2)),
        ("Dense(784→64)", "lightgreen", (0.4, 0.4, 0.3, 0.2)),
        ("ReLU", "orange", (0.8, 0.4, 0.2, 0.2)),
        ("Dense(64→10)", "lightgreen", (1.1, 0.4, 0.3, 0.2)),
        ("Softmax", "pink", (1.5, 0.4, 0.2, 0.2))
    ]

    # Add layers as rectangles + labels
    for (name, color, rect) in layers:
        x, y, w, h = rect
        ax.add_patch(patches.Rectangle((x,y), w,h, color=color))
        ax.text(x+w/2, y+h/2, name, ha="center", va="center", fontsize=12)

    # Add arrows between layers
    x_positions = [0.3, 0.7, 1.0, 1.4]
    for xp in x_positions:
        ax.arrow(xp, 0.5, 0.1, 0, head_width=0.05, head_length=0.03, fc="black", ec="black")

    ax.set_xlim(0, 1.8)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Classical Model Architecture", fontsize=14)
    plt.tight_layout()
    plt.savefig("assets/classical_model_architecture.png", dpi=150)
    plt.close()


def draw_quantum_model_diagram():
    """
    Draws and saves a diagram of the QuantumModel architecture.
    Layers: Preprocessing(784->num_qubits) -> Quantum Layer -> Dense(num_qubits->10) -> Softmax
    """
    fig, ax = plt.subplots(figsize=(10, 3))

    # Preprocess layer
    ax.add_patch(patches.Rectangle((0.1,0.4),0.4,0.2,color="lightblue"))
    ax.text(0.3,0.5,"Preprocess\n(784→num_qubits)",ha="center",va="center",fontsize=12)

    # Quantum layer
    ax.add_patch(patches.Rectangle((0.6,0.4),0.6,0.2,color="lightyellow"))
    ax.text(0.9,0.5,"Quantum\nLayer",ha="center",va="center",fontsize=12)

    # Dense layer
    ax.add_patch(patches.Rectangle((1.3,0.4),0.4,0.2,color="lightgreen"))
    ax.text(1.5,0.5,"Dense\n(num_qubits→10)",ha="center",va="center",fontsize=12)

    # Softmax
    ax.add_patch(patches.Rectangle((1.8,0.4),0.2,0.2,color="pink"))
    ax.text(1.9,0.5,"Softmax",ha="center",va="center",fontsize=12)

    # Arrows
    arrow_positions = [0.5,1.2,1.7]
    for xp in arrow_positions:
        ax.arrow(xp,0.5,0.1,0,head_width=0.05,head_length=0.03,fc="black",ec="black")

    ax.set_xlim(0,2.1)
    ax.set_ylim(0,1)
    ax.axis('off')
    ax.set_title("Quantum Model Architecture", fontsize=14)
    plt.tight_layout()
    plt.savefig("assets/quantum_model_architecture.png", dpi=150)
    plt.close()


def draw_encoding_methods():
    """
    Draw a figure illustrating angle, amplitude, and basis encoding.
    We show minimalistic conceptual diagrams:
      - Angle: Bloch sphere with rotation
      - Amplitude: vector to state mapping
      - Basis: binary to computational basis state
    """
    fig, axes = plt.subplots(1, 3, figsize=(12,4))

    # Angle encoding diagram (Bloch sphere sketch)
    axes[0].set_title("Angle Encoding")
    # Just draw a circle for Bloch sphere approximation
    circle = plt.Circle((0,0), 1, fill=False)
    axes[0].add_artist(circle)
    axes[0].arrow(0,0,0.7,0,head_width=0.05,head_length=0.05,fc="black",ec="black")
    axes[0].arrow(0,0,0,0.7,head_width=0.05,head_length=0.05,fc="black",ec="black")
    axes[0].text(0.5,0.1,"R_X(θ)", ha="center", va="center")
    axes[0].set_xlim(-1.2,1.2)
    axes[0].set_ylim(-1.2,1.2)
    axes[0].axis('off')

    # Amplitude encoding diagram
    axes[1].set_title("Amplitude Encoding")
    # Show a bar representing classical data
    data = [0.4,0.3,0.1,0.2]
    axes[1].bar(range(len(data)), data)
    axes[1].set_xticks(range(len(data)))
    axes[1].set_yticks([])
    axes[1].text(1.5,0.3,"→ Normalize → |ψ> = Σ x_i |i>", ha="center")
    axes[1].set_xlabel("Classical Index")
    axes[1].set_ylabel("Value")
    axes[1].grid(True)

    # Basis encoding diagram
    axes[2].set_title("Basis Encoding")
    # Just show binary vector to a ket
    axes[2].text(0.5,0.6,"x = [1,0,1]", ha="center",va="center",fontsize=12)
    axes[2].text(0.5,0.4,"→ |101>", ha="center",va="center",fontsize=12)
    axes[2].set_xlim(0,1)
    axes[2].set_ylim(0,1)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("assets/encoding_methods.png", dpi=150)
    plt.close()


def draw_quantum_gates():
    """
    Draw a figure showing basic quantum gates (RX, RY, RZ, CNOT) and their matrix forms.
    We'll just write text next to small boxes.
    """
    fig, ax = plt.subplots(figsize=(12,6))
    ax.axis('off')
    ax.set_title("Common Quantum Gates", fontsize=14)

    y_start = 0.8
    spacing = 0.2
    def add_gate(name, matrix_str, y_pos):
        ax.text(0.1, y_pos, name, fontsize=12, ha='left', va='center', fontweight='bold')
        ax.text(0.3, y_pos, matrix_str, fontsize=10, ha='left', va='center', family='monospace')

    RX_matrix = dedent("""\
    R_X(θ) = [[cos(θ/2), -i sin(θ/2)],
              [-i sin(θ/2), cos(θ/2)]]
    """)

    RY_matrix = dedent("""\
    R_Y(θ) = [[cos(θ/2), -sin(θ/2)],
              [sin(θ/2), cos(θ/2)]]
    """)

    RZ_matrix = dedent("""\
    R_Z(θ) = [[e^{-iθ/2}, 0],
              [0, e^{iθ/2}]]
    """)

    CNOT_matrix = dedent("""\
    CNOT = [[1,0,0,0],
            [0,1,0,0],
            [0,0,0,1],
            [0,0,1,0]]
    """)

    add_gate("R_X(θ)", RX_matrix, 0.8)
    add_gate("R_Y(θ)", RY_matrix, 0.6)
    add_gate("R_Z(θ)", RZ_matrix, 0.4)
    add_gate("CNOT", CNOT_matrix, 0.2)

    plt.savefig("assets/quantum_gates.png", dpi=150)
    plt.close()


def generate_markdown_equations():
    """
    Generate a markdown file with LaTeX-formatted math equations and code snippets.
    """
    md_content = dedent(r"""
    # Quantum Data Encoding and Circuits: Math & Code Reference

    This document provides supplementary mathematical formulas and code snippets for the quantum machine learning project.

    ## Mathematical Equations

    ### Angle Encoding
    Map classical data \( x_i \) into qubit rotations:
    \[
    \ket{\psi} = \bigotimes_{i=1}^n R_X\left(\pi \frac{x_i - \min(x)}{\max(x)-\min(x)}\right)\ket{0}
    \]

    ### Amplitude Encoding
    Normalize classical vector \(\mathbf{x}\):
    \[
    \ket{\psi} = \frac{1}{\sqrt{\sum_i x_i^2}} \sum_i x_i \ket{i}
    \]

    ### Basis Encoding
    Direct mapping of binary vector \(\mathbf{b}\):
    \[
    \ket{\psi} = \ket{b_1 b_2 \dots b_n}
    \]

    ### Parameterized Quantum Gates
    \[
    R_X(\theta) = \begin{bmatrix}
    \cos(\theta/2) & -i\sin(\theta/2) \\
    -i\sin(\theta/2) & \cos(\theta/2)
    \end{bmatrix}, \quad
    R_Y(\theta), R_Z(\theta) \text{ similarly defined}
    \]

    CNOT gate:
    \[
    \text{CNOT} = \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 1 & 0
    \end{bmatrix}
    \]

    ## Code Snippets

    ### Classical Model Code (from classical_model.py)
    ```python
    import torch
    import torch.nn as nn

    class ClassicalModel(nn.Module):
        def __init__(self):
            super(ClassicalModel, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(784, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 10)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.softmax(x)
            return x
    ```

    ### Quantum Model Code (from quantum_modelV2.py)
    ```python
    import torch
    import torch.nn as nn
    import pennylane as qml
    from pennylane.qnn.torch import TorchLayer

    def build_quantum_model(param_values):
        num_qubits = param_values["num_qubits"]
        circuit_depth = param_values["circuit_depth"]
        entanglement = param_values["entanglement"]

        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="torch")
        def qnode(inputs, weights):
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
                x = self.preprocess(x)
                x = self.quantum_layer(x)
                x = self.classical_layer(x)
                x = self.softmax(x)
                return x

        return QuantumModel()
    ```
    """)

    with open("assets/quantum_math_and_code.md", "w") as f:
        f.write(md_content)

def main():
    draw_classical_model_diagram()
    draw_quantum_model_diagram()
    draw_encoding_methods()
    draw_quantum_gates()
    generate_markdown_equations()
    print("Visual assets and markdown file generated in 'assets' directory.")

if __name__ == "__main__":
    main()