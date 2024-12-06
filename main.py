# File: main.py
import os
import sys
import argparse
import time
import torch
from utils.config import Config
from utils.data_loader import DataLoader
from models.classical_model import build_classical_model
from models.quantum_model import build_quantum_model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.logger import Logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='classical', choices=['classical', 'quantum'],
                        help='Mode: classical or quantum')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
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

    print(f"Mode: {mode}, Epochs: {epochs}, Batch: {batch_size}, Optimizer: {optimizer}, LR: {learning_rate}")

    data_loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    # Build appropriate model
    if mode == 'classical':
        model = build_classical_model()
    else:
        model = build_quantum_model()
        if model is None:
            print("Quantum model not implemented.")
            sys.exit(1)

    trainer = Trainer(model, (x_train, y_train), (x_test, y_test),
                      mode=mode, epochs=epochs, batch_size=batch_size,
                      optimizer=optimizer, learning_rate=learning_rate,
                      results_dir=results_dir)

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    trainer.evaluate()

    evaluator = Evaluator(model, data=(x_test, y_test), save_dir=results_dir, mode=mode)
    evaluator.evaluate_model()
    evaluator.save_combined_report()
    evaluator.plot_confusion_matrix(normalize=True, save=True)

    logger = Logger(save_dir=results_dir)
    logger.plot_from_logs(mode=mode)

if __name__ == '__main__':
    main()