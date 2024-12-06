# File: main.py
import os
import sys
import argparse
import time
import torch
from utils.config import Config
from utils.data_loader import DataLoader
from models.classical_model import build_classical_model
from models.quantum_modelV2 import build_quantum_model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.logger import Logger

def parse_args():
    parser = argparse.ArgumentParser(description="Quantum-Classical Hybrid Model Training")
    parser.add_argument('--mode', type=str, default='quantum', choices=['classical', 'quantum'],
                        help='Mode: classical or quantum')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    return parser.parse_args()

def main():
    args = parse_args()
    cli_args = {k:v for k,v in vars(args).items() if v is not None}
    config = Config(cli_args=cli_args)

    mode = config.get('mode')
    epochs = config.get('epochs')
    batch_size = config.get('batch_size')
    optimizer = config.get('optimizer')
    learning_rate = config.get('learning_rate')
    results_dir = config.get('results_dir', 'results')

    print("==========================================")
    print(f"Starting {mode.capitalize()} Mode Training")
    print("Data Loading Phase...")
    data_loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()
    print("Data Loaded Successfully.")

    # Build appropriate model
    print("Model Building Phase...")
    if mode == 'classical':
        model = build_classical_model()
        print("Classical Model Built.")
    else:
        model = build_quantum_model()
        if model is None:
            print("Quantum model not implemented.")
            sys.exit(1)
        print("Quantum Model Built.")

    print("Initializing Trainer...")
    trainer = Trainer(model, (x_train, y_train), (x_test, y_test), mode, epochs, batch_size, optimizer, learning_rate, results_dir)

    print("Training Phase...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    elapsed_train = end_time - start_time
    print(f"Training completed in {elapsed_train:.2f} seconds.")

    print("Evaluation Phase...")
    trainer.evaluate()

    print("Evaluator Phase...")
    evaluator = Evaluator(model, data=(x_test, y_test), save_dir=results_dir, mode=mode)
    evaluator.evaluate_model()
    evaluator.save_combined_report()
    evaluator.plot_confusion_matrix(normalize=True, save=True)

    logger = Logger(save_dir=results_dir)
    logger.plot_from_logs(mode=mode)

    # Print a final summary log for video editing convenience
    print("==========================================")
    print(f"[{mode.capitalize()} Mode] Final Summary:")
    logs_path = os.path.join(results_dir, 'logs', f'{mode}_training_logs.json')
    if os.path.exists(logs_path):
        import json
        with open(logs_path, 'r') as f:
            logs = json.load(f)
        final_train_loss = logs['train_loss'][-1]
        final_train_acc = logs['train_accuracy'][-1]*100
        final_val_loss = logs['val_loss'][-1]
        final_val_acc = logs['val_accuracy'][-1]*100
        print(f"Final Train Loss: {final_train_loss:.4f}, Train Acc: {final_train_acc:.2f}%")
        print(f"Final Val Loss: {final_val_loss:.4f}, Val Acc: {final_val_acc:.2f}%")
    else:
        print("No logs found for final summary.")

    print("All tasks completed successfully!")
    print("==========================================")

if __name__ == '__main__':
    main()