import json
import os

class Config:
    def __init__(self, config_path='config.json', cli_args=None):
        self.cli_args = cli_args if cli_args else {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config_data = json.load(f)
        else:
            self.config_data = self.default_config()
        self._apply_cli_overrides()
        self._validate_config()
        self._prepare_directories()

    def default_config(self):
        return {
            "mode": "quantum",
            "epochs": 5,
            "batch_size": 32,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "quantum": {
                "encoding": "angle",
                "num_qubits": 4,
                "circuit_depth": 1,
                "entanglement": "linear",
                "noise_level": 0.0
            },
            "results_dir": "results"
        }

    def _apply_cli_overrides(self):
        if 'mode' in self.cli_args:
            self.config_data['mode'] = self.cli_args['mode']
        if 'epochs' in self.cli_args:
            self.config_data['epochs'] = int(self.cli_args['epochs'])
        if 'batch_size' in self.cli_args:
            self.config_data['batch_size'] = int(self.cli_args['batch_size'])
        if 'optimizer' in self.cli_args:
            self.config_data['optimizer'] = self.cli_args['optimizer']
        if 'learning_rate' in self.cli_args:
            self.config_data['learning_rate'] = float(self.cli_args['learning_rate'])

    def _validate_config(self):
        mode = self.config_data.get('mode', 'classical')
        if mode not in ['classical', 'quantum']:
            raise ValueError("Invalid mode. Must be 'classical' or 'quantum'.")
        epochs = self.config_data.get('epochs', 10)
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        batch_size = self.config_data.get('batch_size', 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        lr = self.config_data.get('learning_rate', 0.001)
        if not isinstance(lr, float) and not isinstance(lr, int):
            raise ValueError("learning_rate must be a float.")
        if lr <= 0:
            raise ValueError("learning_rate must be > 0.")

        qconf = self.config_data.get('quantum', {})
        nq = qconf.get('num_qubits', 4)
        if not isinstance(nq, int) or nq <= 0:
            raise ValueError("Invalid num_qubits.")
        cd = qconf.get('circuit_depth', 1)
        if not isinstance(cd, int) or cd <= 0:
            raise ValueError("Invalid circuit_depth.")
        ent = qconf.get('entanglement', 'linear')
        if ent not in ['linear', 'circular']:
            raise ValueError("Invalid entanglement.")
        enc = qconf.get('encoding', 'angle')
        if enc not in ['angle', 'amplitude', 'basis']:
            raise ValueError("Invalid encoding.")
        noise = qconf.get('noise_level', 0.0)
        if (not isinstance(noise, (float,int))) or noise < 0:
            raise ValueError("Invalid noise_level.")

    def _prepare_directories(self):
        base_dir = self.config_data.get('results_dir', 'results')
        subdirs = ['logs', 'plots', 'raw']
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        for sd in subdirs:
            d = os.path.join(base_dir, sd)
            if not os.path.exists(d):
                os.makedirs(d)

    def get(self, key, default=None):
        return self.config_data.get(key, default)

    def get_quantum_param(self, key, default=None):
        return self.config_data.get('quantum', {}).get(key, default)