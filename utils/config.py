import json
import os

class Config:
    def __init__(self, config_path='config.json'):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config_data = json.load(f)
        else:
            self.config_data = self.default_config()
        self._validate_config()

    def default_config(self):
        return {
            "mode": "classical",
            "epochs": 10,
            "batch_size": 32,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "quantum": {
                "encoding": "angle",       # "angle", "amplitude", or "basis"
                "num_qubits": 4,
                "circuit_depth": 1,
                "entanglement": "linear",  # "linear" or "circular"
                "noise_level": 0.0
            }
        }

    def _validate_config(self):
        qconf = self.config_data.get('quantum', {})
        # Validate num_qubits
        num_qubits = qconf.get('num_qubits', 4)
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("Invalid num_qubits. Must be a positive integer.")
        # Validate circuit_depth
        circuit_depth = qconf.get('circuit_depth', 1)
        if not isinstance(circuit_depth, int) or circuit_depth <= 0:
            raise ValueError("Invalid circuit_depth. Must be a positive integer.")
        # Validate entanglement
        entanglement = qconf.get('entanglement', 'linear')
        if entanglement not in ['linear', 'circular']:
            raise ValueError("Invalid entanglement. Use 'linear' or 'circular'.")
        # Validate encoding
        encoding = qconf.get('encoding', 'angle')
        if encoding not in ['angle', 'amplitude', 'basis']:
            raise ValueError("Invalid encoding. Use 'angle', 'amplitude', or 'basis'.")
        # Validate noise_level
        noise_level = qconf.get('noise_level', 0.0)
        if not (isinstance(noise_level, float) or isinstance(noise_level, int)):
            raise ValueError("Invalid noise_level. Must be a float.")
        if noise_level < 0:
            raise ValueError("noise_level must be >= 0.0")

    def get(self, key, default=None):
        return self.config_data.get(key, default)

    def get_quantum_param(self, key, default=None):
        return self.config_data.get('quantum', {}).get(key, default)