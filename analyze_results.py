# File: analyze_results.py
import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def load_experiment_summary(experiments_dir="results/experiments"):
    summary_path = os.path.join(experiments_dir, "experiment_summary.json")
    if not os.path.exists(summary_path):
        print("[ERROR] No experiment_summary.json found.")
        return []
    with open(summary_path, 'r') as f:
        return json.load(f)

def filter_experiments(experiments, filter_key=None, filter_value=None):
    if filter_key and filter_value is not None:
        # Try to convert filter_value to float or int if possible
        try:
            filter_value = float(filter_value)
            # If it's int convertible, do that:
            if filter_value.is_integer():
                filter_value = int(filter_value)
        except:
            pass
        filtered = [exp for exp in experiments if exp["parameters"].get(filter_key) == filter_value]
        return filtered
    return experiments

def plot_metric_vs_parameter(experiments, metric="val_acc", param="num_qubits", save_dir="results/summary_graphs"):
    os.makedirs(save_dir, exist_ok=True)

    # Extract param-value and metric-value
    data_points = []
    for exp in experiments:
        val = exp["parameters"].get(param)
        metric_val = exp["final_metrics"].get(metric, None)
        if val is not None and metric_val is not None:
            data_points.append((val, metric_val))
    if not data_points:
        print("[INFO] No data points to plot.")
        return
    x_vals = [d[0] for d in data_points]
    y_vals = [d[1] for d in data_points]

    plt.figure(figsize=(8,6))
    plt.scatter(x_vals, y_vals, marker='o')
    plt.xlabel(param)
    plt.ylabel(metric)
    plt.title(f"{metric} vs {param}")
    plt.grid(True)

    plot_path = os.path.join(save_dir, f"{metric}_vs_{param}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Plot saved at {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize experiment results.")
    parser.add_argument('--filter_key', type=str, help='Parameter key to filter by')
    parser.add_argument('--filter_value', help='Value of parameter to filter')
    parser.add_argument('--metric', type=str, default='val_acc', help='Metric to plot (e.g., val_acc, train_acc, val_loss)')
    parser.add_argument('--param', type=str, default='num_qubits', help='Parameter to plot against (e.g., num_qubits)')
    args = parser.parse_args()

    experiments = load_experiment_summary()
    if not experiments:
        return

    filtered_exps = filter_experiments(experiments, args.filter_key, args.filter_value)
    plot_metric_vs_parameter(filtered_exps, metric=args.metric, param=args.param)

if __name__ == '__main__':
    main()