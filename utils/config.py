import json
import os

class Config:
    def __init__(self, config_path='config.json'):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config_data = json.load(f)
        else:
            self.config_data = self.default_config()

    def default_config(self):
        # Default configuration values
        return {
            "mode": "classical",  # "classical" or "quantum" (quantum not implemented yet)
            "epochs": 10,
            "batch_size": 32,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "quantum": {
                "encoding": "angle",  # placeholder for future use
                "num_qubits": 4,
                "circuit_depth": 1,
                "entanglement": "linear"
            }
        }

    def get(self, key, default=None):
        return self.config_data.get(key, default)

    def get_quantum_param(self, key, default=None):
        return self.config_data.get('quantum', {}).get(key, default)