import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_classical_model():
    """
    Manually draws a diagram of the classical model architecture.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add layers
    ax.add_patch(patches.Rectangle((0.1, 0.7), 0.3, 0.15, color="lightblue", label="Flatten"))
    ax.text(0.25, 0.8, "Flatten", ha="center", va="center", fontsize=12)

    ax.add_patch(patches.Rectangle((0.5, 0.7), 0.3, 0.15, color="lightgreen"))
    ax.text(0.65, 0.8, "Dense (784→64)", ha="center", va="center", fontsize=12)

    ax.add_patch(patches.Rectangle((0.9, 0.7), 0.3, 0.15, color="orange"))
    ax.text(1.05, 0.8, "ReLU", ha="center", va="center", fontsize=12)

    ax.add_patch(patches.Rectangle((1.3, 0.7), 0.3, 0.15, color="lightgreen"))
    ax.text(1.45, 0.8, "Dense (64→10)", ha="center", va="center", fontsize=12)

    ax.add_patch(patches.Rectangle((1.7, 0.7), 0.3, 0.15, color="pink"))
    ax.text(1.85, 0.8, "Softmax", ha="center", va="center", fontsize=12)

    # Add arrows
    for x_start, x_end in zip([0.4, 0.8, 1.2, 1.6], [0.5, 0.9, 1.3, 1.7]):
        ax.arrow(x_start, 0.775, 0.1, 0, head_width=0.03, head_length=0.05, fc="black", ec="black")

    ax.set_xlim(0, 2.1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Classical Model Architecture", fontsize=14)

    plt.tight_layout()
    plt.show()


def draw_quantum_model():
    """
    Manually draws a diagram of the quantum model architecture.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Add layers
    ax.add_patch(patches.Rectangle((0.1, 0.7), 0.4, 0.15, color="lightblue"))
    ax.text(0.3, 0.8, "Preprocess (784→num_qubits)", ha="center", va="center", fontsize=12)

    ax.add_patch(patches.Rectangle((0.6, 0.7), 0.6, 0.15, color="lightyellow"))
    ax.text(0.9, 0.8, "Quantum Layer", ha="center", va="center", fontsize=12)

    ax.add_patch(patches.Rectangle((1.3, 0.7), 0.4, 0.15, color="lightgreen"))
    ax.text(1.5, 0.8, "Dense (num_qubits→10)", ha="center", va="center", fontsize=12)

    ax.add_patch(patches.Rectangle((1.8, 0.7), 0.3, 0.15, color="pink"))
    ax.text(1.95, 0.8, "Softmax", ha="center", va="center", fontsize=12)

    # Add arrows
    for x_start, x_end in zip([0.5, 1.2, 1.7], [0.6, 1.3, 1.8]):
        ax.arrow(x_start, 0.775, 0.1, 0, head_width=0.03, head_length=0.05, fc="black", ec="black")

    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Quantum Model Architecture", fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_classical_model()
    draw_quantum_model()