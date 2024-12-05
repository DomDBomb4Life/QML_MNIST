import os
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, save_dir='results'):
        self.history = None
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def set_history(self, history):
        self.history = history

    def plot_training_history(self, save=False):
        if self.history is None:
            print("No training history to plot.")
            return

        # The Keras history object is assumed. If None returned for quantum, handle gracefully.
        if not hasattr(self.history, 'history'):
            print("No valid training history available.")
            return

        epochs = range(1, len(self.history.history['accuracy']) + 1)
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(epochs, self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.history.history['loss'], label='Train Loss')
        plt.plot(epochs, self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.plot(epochs, self.history.history['val_loss'], label='Validation Loss')
        plt.title('Validation Metrics Comparison')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
            plt.close()
        else:
            plt.show()