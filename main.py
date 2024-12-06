import os
import sys
import argparse
import time
from utils.config import Config
from utils.data_loader import DataLoader
from models.classical_model import build_classical_model
from models.quantum_model import build_quantum_model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.logger import Logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='Mode: classical or quantum')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--optimizer', type=str, help='Optimizer type')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
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
    data = ((x_train, y_train), (x_test, y_test))

    if mode == 'classical':
        model = build_classical_model()
    else:
        model = build_quantum_model()
        if model is None:
            print("Quantum model not implemented.")
            sys.exit(1)

    trainer = Trainer(model, data, data_loader, mode, epochs, batch_size, optimizer, learning_rate, results_dir)
    trainer.compile_model()
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