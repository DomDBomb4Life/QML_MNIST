# File: training/trainer.py
import os
import json
import time
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from utils.config import Config

class BaseTrainer:
    """
    Base Trainer class to handle basic training loops for both classical and quantum models.
    """
    def __init__(self, model, train_data, test_data, mode='classical', epochs=10, batch_size=32,
                 optimizer='adam', learning_rate=0.001, results_dir='results'):
        (x_train, y_train) = train_data
        (x_test, y_test) = test_data

        self.model = model
        self.mode = mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_type = optimizer.lower()
        self.learning_rate = learning_rate
        self.results_dir = results_dir
        self.logs_path = os.path.join(self.results_dir, 'logs', f'{self.mode}_training_logs.json')
        self.metrics_history = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.from_numpy(x_train).float()
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.from_numpy(y_train).long()
        if not isinstance(x_test, torch.Tensor):
            x_test = torch.from_numpy(x_test).float()
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.from_numpy(y_test).long()

        self.train_dataset = TensorDataset(x_train, y_train)
        self.test_dataset = TensorDataset(x_test, y_test)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer(self.optimizer_type, self.learning_rate)

        self.config = Config()

    def _get_optimizer(self, optimizer_type, lr):
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _evaluate_dataset(self):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def _save_logs(self):
        os.makedirs(os.path.dirname(self.logs_path), exist_ok=True)
        with open(self.logs_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)

    def evaluate(self):
        # Simple evaluation on test set
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc*100:.2f}%")

class ClassicalTrainer(BaseTrainer):
    def __init__(self, model, train_data, test_data, **kwargs):
        super().__init__(model, train_data, test_data, mode='classical', **kwargs)

    def train(self):
        start = time.time()
        for epoch in range(1, self.epochs+1):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            total_samples = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                total_samples += inputs.size(0)

            avg_train_loss = train_loss / total_samples
            avg_train_acc = train_correct / total_samples

            val_loss, val_acc = self._evaluate_dataset()

            self.metrics_history['epoch'].append(epoch)
            self.metrics_history['train_loss'].append(avg_train_loss)
            self.metrics_history['train_accuracy'].append(avg_train_acc)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['val_accuracy'].append(val_acc)

            elapsed = time.time() - start
            print(f"[Classical Mode] Epoch {epoch}/{self.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                  f"Elapsed: {elapsed:.2f}s")

        self._save_logs()
        # Print summary
        final_train_acc = self.metrics_history['train_accuracy'][-1]*100
        final_train_loss = self.metrics_history['train_loss'][-1]
        final_val_acc = self.metrics_history['val_accuracy'][-1]*100
        final_val_loss = self.metrics_history['val_loss'][-1]
        print("[Classical Mode] Training Summary:")
        print(f"Final Train Loss: {final_train_loss:.4f} | Final Train Acc: {final_train_acc:.2f}%")
        print(f"Final Val Loss: {final_val_loss:.4f} | Final Val Acc: {final_val_acc:.2f}%")

class QuantumTrainer(BaseTrainer):
    def __init__(self, model, train_data, test_data, **kwargs):
        super().__init__(model, train_data, test_data, mode='quantum', **kwargs)
        self.num_qubits = self.config.get_quantum_param('num_qubits', 4)
        self.circuit_depth = self.config.get_quantum_param('circuit_depth', 1)
        self.entanglement = self.config.get_quantum_param('entanglement', 'linear')

    def train(self):
        print(f"[Quantum Mode] Starting training with {self.num_qubits} qubits, "
              f"circuit_depth={self.circuit_depth}, entanglement={self.entanglement}")
        start = time.time()
        for epoch in range(1, self.epochs+1):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            total_samples = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()

                # Compute gradient norms for quantum vs classical parameters
                q_grad_norm = 0.0
                c_grad_norm = 0.0
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        norm = p.grad.data.norm().item()
                        if 'quantum_layer' in name:
                            q_grad_norm += norm
                        else:
                            c_grad_norm += norm

                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                total_samples += inputs.size(0)

            avg_train_loss = train_loss / total_samples
            avg_train_acc = train_correct / total_samples

            val_loss, val_acc = self._evaluate_dataset()

            self.metrics_history['epoch'].append(epoch)
            self.metrics_history['train_loss'].append(avg_train_loss)
            self.metrics_history['train_accuracy'].append(avg_train_acc)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['val_accuracy'].append(val_acc)

            elapsed = time.time() - start
            print(f"[Quantum Mode] Epoch {epoch}/{self.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                  f"Quantum Grad Norm: {q_grad_norm:.4f} | Classical Grad Norm: {c_grad_norm:.4f} | "
                  f"Elapsed: {elapsed:.2f}s")

        self._save_logs()
        # Print summary
        final_train_acc = self.metrics_history['train_accuracy'][-1]*100
        final_train_loss = self.metrics_history['train_loss'][-1]
        final_val_acc = self.metrics_history['val_accuracy'][-1]*100
        final_val_loss = self.metrics_history['val_loss'][-1]
        print("[Quantum Mode] Training Summary:")
        print(f"Final Train Loss: {final_train_loss:.4f} | Final Train Acc: {final_train_acc:.2f}%")
        print(f"Final Val Loss: {final_val_loss:.4f} | Val Acc: {final_val_acc:.2f}%")

def Trainer(model, train_data, test_data, mode='classical', epochs=10, batch_size=32,
            optimizer='adam', learning_rate=0.001, results_dir='results'):
    """
    Factory function to return the appropriate trainer based on mode.
    """
    if mode == 'classical':
        return ClassicalTrainer(model, train_data, test_data, epochs=epochs, batch_size=batch_size,
                                optimizer=optimizer, learning_rate=learning_rate, results_dir=results_dir)
    else:
        return QuantumTrainer(model, train_data, test_data, epochs=epochs, batch_size=batch_size,
                              optimizer=optimizer, learning_rate=learning_rate, results_dir=results_dir)