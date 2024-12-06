import os
import json

class Config:
    """
    Configuration class for managing training parameters.
    """
    def __init__(self, config_path='config.json', cli_args=None):
        self.cli_args = cli_args if cli_args else {}
        
        # Load configuration from file or use defaults
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.default_config()
        
        # Apply CLI argument overrides
        self._apply_cli_overrides()

        # Validate the final configuration
        self._validate_config()

        # Prepare output directories
        self._prepare_directories()

    def default_config(self):
        """
        Default configuration with classical and quantum-specific parameters.
        """
        return {
            "mode": "classical",
            "epochs": 5,
            "batch_size": 32,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "results_dir": "results",
            "quantum": {
                "encoding": "angle",
                "num_qubits": 4,
                "circuit_depth": 1,
                "entanglement": "linear",
                "noise_level": 0.0
            }
        }

    def _apply_cli_overrides(self):
        """
        Override configuration values with CLI arguments if provided.
        """
        for key, value in self.cli_args.items():
            if key in self.config:
                self.config[key] = self._cast_type(value, type(self.config[key]))
            elif key in self.config.get('quantum', {}):
                self.config['quantum'][key] = self._cast_type(value, type(self.config['quantum'][key]))

    def _cast_type(self, value, target_type):
        """
        Cast CLI argument values to the correct type.
        """
        try:
            return target_type(value)
        except (ValueError, TypeError):
            return value

    def _validate_config(self):
        """
        Validate the configuration to ensure all parameters are correct.
        """
        mode = self.config.get('mode')
        if mode not in ['classical', 'quantum']:
            raise ValueError("Invalid mode. Must be 'classical' or 'quantum'.")
        
        epochs = self.config.get('epochs')
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Epochs must be a positive integer.")
        
        batch_size = self.config.get('batch_size')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        
        learning_rate = self.config.get('learning_rate')
        if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
            raise ValueError("Learning rate must be a positive number.")
        
        if mode == 'quantum':
            quantum_config = self.config.get('quantum', {})
            num_qubits = quantum_config.get('num_qubits')
            if not isinstance(num_qubits, int) or num_qubits <= 0:
                raise ValueError("Number of qubits must be a positive integer.")
            
            circuit_depth = quantum_config.get('circuit_depth')
            if not isinstance(circuit_depth, int) or circuit_depth <= 0:
                raise ValueError("Circuit depth must be a positive integer.")
            
            entanglement = quantum_config.get('entanglement')
            if entanglement not in ['linear', 'circular']:
                raise ValueError("Entanglement must be 'linear' or 'circular'.")
            
            encoding = quantum_config.get('encoding')
            if encoding not in ['angle', 'amplitude', 'basis']:
                raise ValueError("Encoding must be 'angle', 'amplitude', or 'basis'.")
            
            noise_level = quantum_config.get('noise_level')
            if not isinstance(noise_level, (float, int)) or noise_level < 0:
                raise ValueError("Noise level must be a non-negative number.")

    def _prepare_directories(self):
        """
        Prepare required output directories for results.
        """
        results_dir = self.config.get('results_dir', 'results')
        subdirs = ['logs', 'plots', 'raw']
        os.makedirs(results_dir, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)

    def get(self, key, default=None):
        """
        Retrieve a value from the configuration.
        """
        return self.config.get(key, default)

    def get_quantum_param(self, key, default=None):
        """
        Retrieve a quantum-specific parameter from the configuration.
        """
        return self.config.get('quantum', {}).get(key, default)