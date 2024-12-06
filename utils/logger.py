# File: utils/logger.py
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

class Logger:
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        self.logs_dir = os.path.join(self.save_dir, 'logs')
        self.plots_dir = os.path.join(self.save_dir, 'plots')
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_from_logs(self, mode='classical'):
        logs_path = os.path.join(self.logs_dir, f'{mode}_training_logs.json')
        if not os.path.exists(logs_path):
            print(f"No logs found at {logs_path}")
            return
        
        with open(logs_path, 'r') as f:
            logs = json.load(f)
        
        epochs = logs['epoch']
        train_loss = logs['train_loss']
        train_acc = logs['train_accuracy']
        val_loss = logs['val_loss']
        val_acc = logs['val_accuracy']

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label='Train Loss', marker='o')
        plt.plot(epochs, val_loss, label='Val Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{mode.capitalize()} Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, label='Train Acc', marker='o')
        plt.plot(epochs, val_acc, label='Val Acc', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{mode.capitalize()} Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.plots_dir, f'{mode}_training_metrics.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()