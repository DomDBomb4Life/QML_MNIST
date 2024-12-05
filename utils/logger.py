import matplotlib.pyplot as plt
import os

class Logger:
    def __init__(self, save_dir='results'):
        self.history = None
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def set_history(self, history):
        self.history = history

    def plot_training_history(self):
        if self.history is None:
            print("No training history.")
            return

        plt.figure(figsize=(12,5))
        epochs = range(1, len(self.history.history['accuracy'])+1)

        plt.subplot(1,2,1)
        plt.plot(epochs, self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(epochs, self.history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(epochs, self.history.history['loss'], label='Train Loss')
        plt.plot(epochs, self.history.history['val_loss'], label='Val Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()