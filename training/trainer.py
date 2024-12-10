# File: training/trainer.py
import os
import json
import time
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class BaseTrainer:
    """
    Base Trainer class. Does not rely on Config().
    All parameters come from constructor arguments.
    """
    def __init__(self, model, train_data, test_data, mode='classical', epochs=10, batch_size=32,
                 optimizer='adam', learning_rate=0.001, results_dir='results',
                 log_every_batch=False, track_gradients=False, track_time=False):
        self.model = model
        self.mode = mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_type = optimizer.lower()
        self.learning_rate = learning_rate
        self.results_dir = results_dir
        self.log_every_batch = log_every_batch
        self.track_gradients = track_gradients
        self.track_time = track_time

        (x_train, y_train) = train_data
        (x_test, y_test) = test_data

        # Convert data to tensors if needed
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
        self.optimizer = self._get_optimizer()

        # Prepare logs directory
        logs_dir = os.path.join(self.results_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.logs_path = os.path.join(logs_dir, f"{self.mode}_training_logs.json")

        # If logs exist, load them; else initialize
        if os.path.exists(self.logs_path):
            with open(self.logs_path, 'r') as f:
                self.logs = json.load(f)
        else:
            self.logs = {
                "epoch": [],
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": [],
                "batch_metrics": []
            }

    def _get_optimizer(self):
        if self.optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

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
        with open(self.logs_path, 'w') as f:
            json.dump(self.logs, f, indent=4)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            total_samples = 0
            epoch_start_time = time.time() if self.track_time else None

            epoch_batch_metrics = []  # store batch-level metrics if enabled

            for batch_i, (inputs, labels) in enumerate(self.train_loader, start=1):
                batch_start_time = time.time() if self.track_time else None
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                if (batch_i % 100) == 0:
                    print("Batch: ", batch_i)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()

                # Gradient norms if track_gradients=True
                q_grad_norm = 0.0
                c_grad_norm = 0.0
                if self.track_gradients:
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            norm = p.grad.data.norm().item()
                            if "quantum_layer" in name:
                                q_grad_norm += norm
                            else:
                                c_grad_norm += norm

                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                total_samples += inputs.size(0)

                if self.log_every_batch:
                    batch_entry = {
                        "epoch": epoch,
                        "batch": batch_i,
                        "batch_loss": loss.item(),
                        "batch_accuracy": (preds == labels).sum().item() / inputs.size(0),
                        "batch_time": (time.time() - batch_start_time) if self.track_time else None,
                        "quantum_grad_norm": q_grad_norm if self.track_gradients else None,
                        "classical_grad_norm": c_grad_norm if self.track_gradients else None
                    }
                    epoch_batch_metrics.append(batch_entry)

            avg_train_loss = train_loss / total_samples
            avg_train_acc = train_correct / total_samples
            val_loss, val_acc = self._evaluate_dataset()

            self.logs["epoch"].append(epoch)
            self.logs["train_loss"].append(avg_train_loss)
            self.logs["train_accuracy"].append(avg_train_acc)
            self.logs["val_loss"].append(val_loss)
            self.logs["val_accuracy"].append(val_acc)
            if self.log_every_batch:
                self.logs["batch_metrics"].extend(epoch_batch_metrics)

            epoch_time = time.time() - epoch_start_time if self.track_time else None
            print(f"Epoch {epoch}/{self.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
                  f"{' | Epoch Time: {:.2f}s'.format(epoch_time) if self.track_time else ''}")

        self._save_logs()

    def evaluate(self):
        val_loss, val_acc = self._evaluate_dataset()
        print(f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc*100:.2f}%")


class ClassicalTrainer(BaseTrainer):
    pass


class QuantumTrainer(BaseTrainer):
    pass