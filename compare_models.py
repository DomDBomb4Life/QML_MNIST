import os
import json
import matplotlib.pyplot as plt


def load_logs(logs_path):
    """Load training logs from the given path."""
    if not os.path.exists(logs_path):
        print(f"[WARNING] No logs found at {logs_path}. Skipping.")
        return None
    with open(logs_path, 'r') as f:
        return json.load(f)


def plot_comparison(classical_logs, quantum_logs, results_dir):
    """Plot training and validation accuracy for both models."""
    if not classical_logs or not quantum_logs:
        print("[ERROR] Missing logs for one or both models. Cannot plot.")
        return

    # Extract data for plotting
    classical_epochs = classical_logs["epoch"]
    classical_train_acc = classical_logs["train_accuracy"]
    classical_val_acc = classical_logs["val_accuracy"]

    quantum_epochs = quantum_logs["epoch"]
    quantum_train_acc = quantum_logs["train_accuracy"]
    quantum_val_acc = quantum_logs["val_accuracy"]

    # Create figure with two subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot for training accuracy
    axes[0].plot(classical_epochs, classical_train_acc, label="Classical", color="blue", marker="o")
    axes[0].plot(quantum_epochs, quantum_train_acc, label="Quantum", color="green", marker="x")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training Accuracy")
    axes[0].legend()
    axes[0].grid()

    # Subplot for validation accuracy
    axes[1].plot(classical_epochs, classical_val_acc, label="Classical", color="blue", marker="o")
    axes[1].plot(quantum_epochs, quantum_val_acc, label="Quantum", color="green", marker="x")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid()

    # Save combined plot
    combined_plot_path = os.path.join(results_dir, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(combined_plot_path)
    plt.close()
    print(f"[INFO] Comparison plot saved at: {combined_plot_path}")


def main():
    results_dir = "results"

    # Paths to the logs for classical and quantum models
    classical_logs_path = os.path.join(results_dir, "classical", "logs", "classical_training_logs.json")
    quantum_logs_path = os.path.join(results_dir, "quantum", "logs", "quantum_training_logs.json")

    # Load logs
    classical_logs = load_logs(classical_logs_path)
    quantum_logs = load_logs(quantum_logs_path)

    # Plot comparison
    plot_comparison(classical_logs, quantum_logs, results_dir)


if __name__ == "__main__":
    main()
