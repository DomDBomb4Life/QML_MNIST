# File: main.py
import os
import json
import time
import torch
from utils.data_loader import DataLoader
from models.classical_model import build_classical_model
from models.quantum_modelV2 import build_quantum_model
from training.trainer import QuantumTrainer as Trainer
from utils.config import Config


def main():
    # =============== User Inputs ===============
    print("Welcome to the Training Script!")
    while True:
        mode = input("Select model mode (classical/quantum): ").strip().lower()
        if mode in ['classical', 'quantum']:
            break
        print("Invalid input. Please type 'classical' or 'quantum'.")

    while True:
        choice = input("Delete existing model and logs? (y/n): ").strip().lower()
        if choice in ['y', 'n']:
            break
        print("Invalid input. Please type 'y' or 'n'.")

    while True:
        epochs_str = input("Enter number of epochs to train: ").strip()
        if epochs_str.isdigit() and int(epochs_str) > 0:
            new_epochs = int(epochs_str)
            break
        print("Invalid input. Please enter a positive integer.")

    # =============== Configuration ===============
    config_dir = f"config/{mode}"  # Separate configs for classical/quantum
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.json")

    if choice == 'y' or not os.path.exists(config_path):
        # Generate a new configuration
        base_config = Config()
        base_config.config["mode"] = mode
        base_config.config["epochs"] = new_epochs

        if mode == "quantum":
            base_config.config["quantum"] = {
                "encoding": "amplitude",
                "num_qubits": 10,
                "circuit_depth": 1,
                "entanglement": "circular",
                "noise_level": 0.3
            }

        # Save config
        with open(config_path, 'w') as f:
            json.dump(base_config.config, f, indent=4)
        print(f"[INFO] New config saved at {config_path}")

    else:
        # Load existing configuration
        base_config = Config(config_path=config_path)
        print(f"[INFO] Loaded existing config from {config_path}")

    # =============== Model and Logs Management ===============
    results_dir = base_config.get('results_dir', 'results')
    mode_dir = os.path.join(results_dir, mode)
    logs_dir = os.path.join(mode_dir, "logs")
    model_path = os.path.join(mode_dir, "trained_model.pth")
    logs_path = os.path.join(logs_dir, f"{mode}_training_logs.json")

    os.makedirs(mode_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    if choice == 'y':
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(logs_path):
            os.remove(logs_path)
        old_logs = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        last_epoch_done = 0
    else:
        if os.path.exists(logs_path):
            with open(logs_path, 'r') as f:
                old_logs = json.load(f)
            last_epoch_done = old_logs["epoch"][-1] if old_logs["epoch"] else 0
        else:
            old_logs = {
                "epoch": [],
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": []
            }
            last_epoch_done = 0

    # =============== Data Loading ===============
    data_loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    # =============== Model Building ===============
    if mode == 'classical':
        model = build_classical_model()
    else:
        param_values = base_config.get('quantum')
        model = build_quantum_model(param_values)

    if choice == 'n' and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    # =============== Trainer Initialization ===============
    trainer = Trainer(
        model=model,
        train_data=(x_train, y_train),
        test_data=(x_test, y_test),
        mode=mode,
        epochs=new_epochs,
        batch_size=base_config.get('batch_size'),
        optimizer=base_config.get('optimizer'),
        learning_rate=base_config.get('learning_rate'),
        results_dir=mode_dir
    )

    trainer.logs = old_logs

    # =============== Training ===============
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Adjust epoch numbering
    trainer.logs["epoch"][-new_epochs:] = [e + last_epoch_done for e in trainer.logs["epoch"][-new_epochs:]]

    # =============== Save Model and Logs ===============
    torch.save(model.state_dict(), model_path)
    trainer._save_logs()

    # =============== Final Summary ===============
    final_train_loss = trainer.logs['train_loss'][-1]
    final_train_acc = trainer.logs['train_accuracy'][-1] * 100
    final_val_loss = trainer.logs['val_loss'][-1]
    final_val_acc = trainer.logs['val_accuracy'][-1] * 100
    print(f"Final Train Loss: {final_train_loss:.4f}, Train Acc: {final_train_acc:.2f}%")
    print(f"Final Val Loss: {final_val_loss:.4f}, Val Acc: {final_val_acc:.2f}%")
    print("Training completed successfully.")


if __name__ == '__main__':
    main()