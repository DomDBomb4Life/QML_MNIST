import os
import time
from utils.config import Config
from utils.data_loader import DataLoader
from models.classical_model import build_classical_model
from models.quantum_model import build_quantum_model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.logger import Logger

def main():
    config = Config()
    mode = config.get('mode', 'classical')
    epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 32)
    optimizer = config.get('optimizer', 'adam')
    learning_rate = config.get('learning_rate', 0.001)

    print("MNIST Digit Recognition - Training Pipeline")
    print("-------------------------------------------")
    print(f"Mode: {mode} | Epochs: {epochs} | Batch Size: {batch_size} | Optimizer: {optimizer} | LR: {learning_rate}")

    data_loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()
    data = ((x_train, y_train), (x_test, y_test))

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Build the appropriate model based on mode
    if mode == 'classical':
        model = build_classical_model()
    else:
        # Placeholder: once quantum is implemented, build_quantum_model() will return a hybrid model
        model = build_quantum_model()
        if model is None:
            print("Quantum model is not implemented yet. Exiting.")
            return

    trainer = Trainer(
        model=model,
        data=data,
        data_loader=data_loader,
        mode=mode,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        learning_rate=learning_rate
    )
    trainer.compile_model()
    start_time = time.time()
    history = trainer.train()
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training Time: {training_time:.2f} seconds")
    trainer.evaluate()

    # Evaluate and log results if we have a classical model and history
    if mode == 'classical' and history is not None:
        evaluator = Evaluator(model, data=(x_test, y_test), save_dir=results_dir)
        evaluator.generate_classification_report()
        evaluator.plot_confusion_matrix(save=True)

        logger = Logger(save_dir=results_dir)
        logger.set_history(history)
        logger.plot_training_history(save=True)

if __name__ == '__main__':
    main()