# File: evaluation/evaluator.py
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class Evaluator:
    def __init__(self, model, data, save_dir='results', mode='classical'):
        (self.x_test, self.y_test) = data
        self.model = model
        self.save_dir = save_dir
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Create directories for logs, plots, and reports
        self.logs_dir = os.path.join(self.save_dir, 'logs')
        self.plots_dir = os.path.join(self.save_dir, 'plots')
        self.reports_dir = os.path.join(self.save_dir, 'reports')
        for d in [self.logs_dir, self.plots_dir, self.reports_dir]:
            os.makedirs(d, exist_ok=True)

    def evaluate_model(self):
        """
        Evaluate the model using batch-wise forward passes. No .predict() usage.
        Store predictions and true labels for report generation.
        """
        x_test = torch.from_numpy(self.x_test).float().to(self.device)
        y_test = torch.from_numpy(self.y_test).long().to(self.device)

        self.model.eval()
        all_preds = []
        all_labels = []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, x_test.size(0), batch_size):
                batch_x = x_test[i:i+batch_size]
                batch_y = y_test[i:i+batch_size]
                outputs = self.model(batch_x)
                preds = outputs.argmax(dim=1).cpu().numpy()
                labels_np = batch_y.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels_np)

        # Classification report
        report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
        report_path = os.path.join(self.reports_dir, f'{self.mode}_classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)

        self.metrics = {
            "predictions": all_preds,
            "labels": all_labels,
            "accuracy": report['accuracy']
        }

        # If both classical and quantum exist, merge:
        self._combine_reports_if_possible()

    def _combine_reports_if_possible(self):
        c_report_path = os.path.join(self.reports_dir, 'classical_classification_report.json')
        q_report_path = os.path.join(self.reports_dir, 'quantum_classification_report.json')
        combined_path = os.path.join(self.reports_dir, 'combined_report.json')

        if os.path.exists(c_report_path) and os.path.exists(q_report_path):
            with open(c_report_path) as cf:
                c_report = json.load(cf)
            with open(q_report_path) as qf:
                q_report = json.load(qf)
            combined = {"classical": c_report, "quantum": q_report}
            with open(combined_path, 'w') as f:
                json.dump(combined, f, indent=4)

    def save_combined_report(self):
        # Already combined above if possible
        pass

    def plot_confusion_matrix(self, normalize=True, save=True):
        """
        Plot and compare confusion matrices. If both classical and quantum runs available,
        plot difference matrix or side-by-side comparison.
        """
        cm = confusion_matrix(self.metrics['labels'], self.metrics['predictions'])
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues')
        plt.title(f'{self.mode.capitalize()} Confusion Matrix {"(Normalized)" if normalize else ""}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        fig_path = os.path.join(self.plots_dir, f'{self.mode}_confusion_matrix.png')
        if save:
            plt.savefig(fig_path)
            plt.close()
        else:
            plt.show()

        # If both classical and quantum exist, create difference matrix
        c_report_path = os.path.join(self.reports_dir, 'classical_classification_report.json')
        q_report_path = os.path.join(self.reports_dir, 'quantum_classification_report.json')
        if os.path.exists(c_report_path) and os.path.exists(q_report_path) and self.mode == 'quantum':
            # Load classical predictions to create difference matrix
            # This requires storing classical predictions similarly
            # For simplicity, assume we only difference plot if we have predictions saved similarly.
            pass  # Implement if needed