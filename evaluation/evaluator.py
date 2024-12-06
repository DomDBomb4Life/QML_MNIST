import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class Evaluator:
    def __init__(self, model, data, save_dir='results', mode='classical'):
        (self.x_test, self.y_test) = data
        self.model = model
        self.save_dir = save_dir
        self.mode = mode
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def evaluate_model(self):
        y_pred = self.model.predict(self.x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        report = classification_report(y_true, y_pred_classes, digits=4, output_dict=True)
        with open(os.path.join(self.save_dir, f'{self.mode}_classification_report.json'), 'w') as f:
            json.dump(report, f, indent=4)

        if self.mode == 'quantum':
            qubit_analysis = self._quantum_per_qubit_analysis(y_pred)
            with open(os.path.join(self.save_dir, 'quantum_qubit_analysis.json'), 'w') as f:
                json.dump(qubit_analysis, f, indent=4)

    def _quantum_per_qubit_analysis(self, y_pred):
        # Placeholder for more detailed quantum analysis
        # Here we just return averages
        return {"avg_outputs_per_qubit": y_pred.mean(axis=0).tolist()}

    def save_combined_report(self):
        # Placeholder: If we had multiple runs (classical & quantum), combine them
        # For now, just check if both classical and quantum reports exist and merge
        base = self.save_dir
        c_report_path = os.path.join(base, 'classical_classification_report.json')
        q_report_path = os.path.join(base, 'quantum_classification_report.json')
        if os.path.exists(c_report_path) and os.path.exists(q_report_path):
            with open(c_report_path) as f:
                c_report = json.load(f)
            with open(q_report_path) as f:
                q_report = json.load(f)
            combined = {"classical": c_report, "quantum": q_report}
            with open(os.path.join(base, 'combined_report.json'), 'w') as f:
                json.dump(combined, f, indent=4)

    def plot_confusion_matrix(self, normalize=False, save=False):
        y_pred = self.model.predict(self.x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred_classes, normalize='true' if normalize else None)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues')
        plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(np.arange(10)+0.5, np.arange(10))
        plt.yticks(np.arange(10)+0.5, np.arange(10), rotation=0)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_dir, f'{self.mode}_confusion_matrix.png'))
            plt.close()
        else:
            plt.show()