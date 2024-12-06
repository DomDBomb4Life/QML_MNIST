# File: run_experiments.py
import os
import json
import itertools
import time
import torch

from utils.config import Config
from utils.data_loader import DataLoader
from models.quantum_modelV2 import build_quantum_model
from training.trainer import QuantumTrainer
from utils.hash_generator import generate_config_hash


def main():
    # param_grid = {
    #     "num_qubits": [2, 4, 8],
    #     "circuit_depth": [1, 2],
    #     "entanglement": ["linear", "circular"],
    #     "encoding": ["angle", "amplitude"],
    #     "noise_level": [0.0, 0.1]
    # }
    param_grid = {
        "num_qubits": [2, 4, 8, 16, 32]
    }

    results_dir = "results"
    experiments_dir = os.path.join(results_dir, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)

    base_config = Config()
    base_config.config["mode"] = "quantum"
    base_config.config["epochs"] = 5
    base_config.config["batch_size"] = 32
    base_config.config["optimizer"] = "adam"
    base_config.config["learning_rate"] = 0.001

    keys = list(param_grid.keys())
    param_combinations = list(itertools.product(*[param_grid[k] for k in keys]))

    data_loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    summary_path = os.path.join(experiments_dir, "experiment_summary.json")
    experiment_summaries = []
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            experiment_summaries = json.load(f)

    existing_hashes = {exp["id"]: exp for exp in experiment_summaries}

    for combo in param_combinations:
        param_values = dict(zip(keys, combo))
        for key, value in param_values.items():
            base_config.config["quantum"][key] = value

        hash_id = generate_config_hash(param_values)
        config_dir = os.path.join(experiments_dir, hash_id)
        os.makedirs(config_dir, exist_ok=True)

        print(f"\n[INFO] Running Experiment {hash_id} with Parameters:")
        for key, value in param_values.items():
            print(f"  {key}: {value}")

        config_path = os.path.join(config_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump({"parameters": param_values}, f, indent=4)

        model = build_quantum_model()

        trainer = QuantumTrainer(
            model=model,
            train_data=(x_train, y_train),
            test_data=(x_test, y_test),
            mode="quantum",
            epochs=base_config.config["epochs"],
            batch_size=base_config.config["batch_size"],
            optimizer=base_config.config["optimizer"],
            learning_rate=base_config.config["learning_rate"],
            results_dir=config_dir,
            log_every_batch=True,
            track_gradients=True,
            track_time=True
        )

        start_time = time.time()
        trainer.train()
        trainer.evaluate()
        elapsed_time = time.time() - start_time

        logs_path = os.path.join(config_dir, "logs", "quantum_training_logs.json")
        if os.path.exists(logs_path):
            with open(logs_path, 'r') as f:
                logs = json.load(f)
            final_train_loss = logs["train_loss"][-1]
            final_train_acc = logs["train_accuracy"][-1] * 100
            final_val_loss = logs["val_loss"][-1]
            final_val_acc = logs["val_accuracy"][-1] * 100
        else:
            final_train_loss = final_train_acc = final_val_loss = final_val_acc = None

        model_path = os.path.join(config_dir, "model.pth")
        torch.save(model.state_dict(), model_path)

        experiment_summary = {
            "id": hash_id,
            "parameters": param_values,
            "final_metrics": {
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
                "train_acc": final_train_acc,
                "val_acc": final_val_acc,
                "runtime": elapsed_time
            },
            "paths": {
                "config": config_path,
                "logs": logs_path,
                "model": model_path
            }
        }

        if hash_id in existing_hashes:
            existing_hashes[hash_id].update(experiment_summary)
        else:
            experiment_summaries.append(experiment_summary)

        print(f"[INFO] Completed Experiment {hash_id}: "
              f"Train Loss={final_train_loss}, Val Acc={final_val_acc:.2f}%, Runtime={elapsed_time:.2f}s")

    with open(summary_path, 'w') as f:
        json.dump(experiment_summaries, f, indent=4)

    print("\n[INFO] All experiments completed. Summary saved at:", summary_path)


if __name__ == '__main__':
    main()