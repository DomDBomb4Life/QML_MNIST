import sys
from data_loader import DataLoader
from models.classical_model import build_classical_model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.logger import Logger

def main():
    print("MNIST Digit Recognition")
    print("-----------------------")
    mode = input("Select mode (classical/quantum): ").strip().lower()
    epochs = input("Enter number of training epochs (e.g., 10): ").strip()
    if not epochs.isdigit():
        print("Invalid number of epochs. Using default of 10.")
        epochs = 10
    else:
        epochs = int(epochs)

    data_loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    if mode == 'classical':
        print("Training Classical Model...")
        model = build_classical_model()
    elif mode == 'quantum':
        print("Quantum mode selected, but quantum model is not implemented yet.")
        sys.exit(0)
    else:
        print("Invalid mode selected. Defaulting to classical mode.")
        model = build_classical_model()

    trainer = Trainer(model, data=((x_train, y_train), (x_test, y_test)), data_loader=data_loader, epochs=epochs)
    trainer.compile_model()
    history = trainer.train()

    trainer.evaluate()

    evaluator = Evaluator(model, data=(x_test, y_test))
    evaluator.generate_classification_report()
    evaluator.plot_confusion_matrix()

    logger = Logger()
    logger.set_history(history)
    logger.plot_training_history()

    # Option to save the model and plots
    save_option = input("Do you want to save the trained model and plots? (yes/no): ").strip().lower()
    if save_option == 'yes':
        model.save('trained_model.h5')
        logger.save_plots()
        evaluator.save_confusion_matrix()
        print("Model and plots have been saved.")

if __name__ == '__main__':
    main()