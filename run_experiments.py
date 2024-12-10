import os
import json
import time
import torch
import matplotlib.pyplot as plt

from utils.config import Config
from utils.data_loader import DataLoader
from models.quantum_modelV2 import build_quantum_model
from training.trainer import QuantumTrainer


def _create_experiment_dir_name(param_name, param_value):
    """Create a directory name based on the primary parameter being varied."""
    return f"{param_name}_{param_value}"


def main(primary_param, values):
    # Define the primary hyperparameter and its range of values
    primary_param = "noise_level"  # Change this to focus on a different hyperparameter
    param_grid = {primary_param: values}

    results_dir = "results"
    experiments_dir = os.path.join(results_dir, "experiments/"+primary_param)
    os.makedirs(experiments_dir, exist_ok=True)

    base_config = Config()
    base_config.config["mode"] = "quantum"
    base_config.config["epochs"] = 5
    base_config.config["batch_size"] = 32
    base_config.config["optimizer"] = "adam"
    base_config.config["learning_rate"] = 0.001

    data_loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    for param_value in param_grid[primary_param]:
        param_values = {
            "num_qubits": 10,
            "circuit_depth": 1,
            "entanglement": "linear",
            "encoding": "amplitude",
            "noise_level": 0.0,
            primary_param: param_value,

        }

        # Create a directory name based on the primary parameter
        dir_name = _create_experiment_dir_name(primary_param, param_value)
        config_dir = os.path.join(experiments_dir, dir_name)
        os.makedirs(config_dir, exist_ok=True)

        print(f"[INFO] Running Experiment {dir_name} with {param_values}")

        config_path = os.path.join(config_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"parameters": param_values}, f, indent=4)

        model = build_quantum_model(param_values)

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
        )

        trainer.train()
        trainer.evaluate()

        print(f"[INFO] Completed Experiment: {dir_name}")


if __name__ == "__main__":
    param = "noise_level"
    values = [0.0, 0.1, 0.2, 0.3, 0.4]
    main(param, values)