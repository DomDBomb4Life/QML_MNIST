import os
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, save_dir='results'):
        self.history = None
        self.save_dir = save_dir

    def set_history(self, history):
        self.history = history

    def plot_training_history(self, save=False):
        if self.history is None:
            print("No training history to plot.")
            return

        epochs = range(1, len(self.history.history['accuracy']) + 1)

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(epochs, self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history.history['loss'], label='Train Loss')
        plt.plot(epochs, self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
            plt.close()
        else:
            plt.show()

    @staticmethod
    def compare_histories(histories, save_dir='results'):
        epochs = range(1, len(next(iter(histories.values()))['history']['accuracy']) + 1)
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        for mode, data in histories.items():
            plt.plot(epochs, data['history']['accuracy'], label=f'{mode.capitalize()} Train Acc')
            plt.plot(epochs, data['history']['val_accuracy'], linestyle='--', label=f'{mode.capitalize()} Val Acc')
        plt.title('Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        training_times = [data['training_time'] for data in histories.values()]
        modes = [mode.capitalize() for mode in histories.keys()]
        plt.bar(modes, training_times)
        plt.title('Training Time Comparison')
        plt.ylabel('Time (seconds)')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'comparison.png'))
        plt.close()

    def save_plots(self):
        self.plot_training_history(save=True)