import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

class Evaluator:
    def __init__(self, model, data, save_dir='results'):
        (self.x_test, self.y_test) = data
        self.model = model
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def generate_classification_report(self):
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        report = classification_report(y_true, y_pred_classes, digits=4)
        print("Classification Report:")
        print(report)
        with open(os.path.join(self.save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)

    def plot_confusion_matrix(self):
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8,6))
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.title('Confusion Matrix')
        plt.colorbar()
        ticks = np.arange(10)
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.close()