import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class Evaluator:
    def __init__(self, model, data, save_dir='results'):
        self.model = model
        (self.x_test, self.y_test) = data
        self.save_dir = save_dir

    def generate_classification_report(self):
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        report = classification_report(y_true, y_pred_classes, digits=4)
        print("Classification Report:")
        print(report)
        # Save report to a text file
        report_path = os.path.join(self.save_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)

    def plot_confusion_matrix(self, normalize=False, save=False):
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred_classes, normalize='true' if normalize else None)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues')
        plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(np.arange(10) + 0.5, labels=np.arange(10))
        plt.yticks(np.arange(10) + 0.5, labels=np.arange(10), rotation=0)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
            plt.close()
        else:
            plt.show()