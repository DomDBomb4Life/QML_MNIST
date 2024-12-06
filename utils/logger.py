import os
import json
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def plot_from_logs(self, mode='classical'):
        log_path = os.path.join(self.save_dir, 'logs', f'{mode}_training_logs.json')
        if not os.path.exists(log_path):
            return
        with open(log_path, 'r') as f:
            history = json.load(f)

        if 'epoch' not in history:
            return

        epochs = history['epoch']
        if len(epochs) == 0:
            return

        acc = history.get('train_accuracy', [])
        val_acc = history.get('val_accuracy', [])
        loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, acc, label='Train Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title('Validation Metrics Comparison')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', f'{mode}_training_history.png'))
        plt.close()