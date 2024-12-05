import os
from data_loader import DataLoader
from models.quantum_model import build_quantum_model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.logger import Logger
from utils.config import Config

def main():
    # Load data and encode
    data_loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    # Build quantum model
    model = build_quantum_model()

    # Train model
    trainer = Trainer(model, data=((x_train, y_train), (x_test, y_test)))
    trainer.compile_model()
    history = trainer.train()

    # Evaluate model
    trainer.evaluate()

    # Evaluate and visualize
    evaluator = Evaluator(model, data=(x_test, y_test), save_dir=Config.RESULTS_DIR)
    evaluator.generate_classification_report()
    evaluator.plot_confusion_matrix()

    # Log results
    logger = Logger(save_dir=Config.RESULTS_DIR)
    logger.set_history(history)
    logger.plot_training_history()
    print("Training complete. Results saved in:", Config.RESULTS_DIR)

if __name__ == '__main__':
    main()