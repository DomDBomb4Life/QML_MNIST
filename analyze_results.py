import os
import json
import matplotlib.pyplot as plt

params = ["num_qubits", "circuit_depth", "encoding", "entanglement", "noise_level"]
def main(param):
    results_dir = "results"
  
    experiments_dir = os.path.join(results_dir, "experiments/"+param)

    if not os.path.exists(experiments_dir):
        print("[ERROR] No experiments directory found. Run experiments first.")
        return

    all_train_curves = []
    all_val_curves = []

    for exp_dir in sorted(os.listdir(experiments_dir)):
        exp_path = os.path.join(experiments_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue

        logs_path = os.path.join(exp_path, "logs", "quantum_training_logs.json")
        if not os.path.exists(logs_path):
            print(f"[WARNING] No logs found for {exp_dir}. Skipping.")
            continue

        with open(logs_path, "r") as f:
            logs = json.load(f)

        epochs = logs["epoch"]
        train_accuracy = logs["train_accuracy"]
        val_accuracy = logs["val_accuracy"]

        all_train_curves.append((exp_dir, epochs, train_accuracy))
        all_val_curves.append((exp_dir, epochs, val_accuracy))

    if not all_train_curves:
        print("[INFO] No valid logs found. Cannot plot.")
        return

    # Create a single figure with two subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot for training accuracy
    for exp_dir, epochs, train_accuracy in all_train_curves:
        axes[0].plot(epochs, train_accuracy, label=exp_dir)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training Accuracy")
    axes[0].legend()
    axes[0].grid()

    # Subplot for validation accuracy
    for exp_dir, epochs, val_accuracy in all_val_curves:
        axes[1].plot(epochs, val_accuracy, label=exp_dir)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid()

    # Save combined plot
    combined_plot_path = os.path.join(results_dir, param+"_graph.png")
    plt.tight_layout()
    plt.savefig(combined_plot_path)
    plt.close()

    print(f"[INFO] Combined training and validation accuracy plot saved at: {combined_plot_path}")


if __name__ == "__main__":
    main("noise_level")
    # for param in params:
    #     main(param)