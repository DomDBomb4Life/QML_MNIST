import os
import sys
import time
import tensorflow as tf
from data_loader import DataLoader
from models.classical_model import build_classical_model
from models.quantum_model import build_quantum_model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.logger import Logger
from utils.image_uploader import ImageUploader

def main():
    print("MNIST Digit Recognition")
    print("-----------------------")
    mode = input("Select mode (classical/quantum/both): ").strip().lower()
    epochs = input("Enter number of training epochs (e.g., 10): ").strip()
    if not epochs.isdigit():
        print("Invalid number of epochs. Using default of 10.")
        epochs = 10
    else:
        epochs = int(epochs)

    data_loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    modes = ['classical', 'quantum'] if mode == 'both' else [mode]
    all_histories = {}

    for current_mode in modes:
        start_time = time.time()
        print(f"\nTraining {current_mode.capitalize()} Model...")
        if current_mode == 'classical':
            model = build_classical_model()
            data = ((x_train, y_train), (x_test, y_test))
        elif current_mode == 'quantum':
            model = build_quantum_model()
            # Reduced dataset for quantum mode
            data = ((x_train[:1000], y_train[:1000]), (x_test[:200], y_test[:200]))
        else:
            print(f"Invalid mode '{current_mode}' selected. Skipping.")
            continue

        trainer = Trainer(
            model,
            data=data,
            data_loader=data_loader,
            epochs=epochs,
            mode=current_mode
        )
        trainer.compile_model()
        history = trainer.train()

        end_time = time.time()
        training_time = end_time - start_time
        print(f"{current_mode.capitalize()} Model Training Time: {training_time:.2f} seconds")

        # Save results
        model_dir = os.path.join(results_dir, current_mode)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model.save(os.path.join(model_dir, 'trained_model.h5'))

        evaluator = Evaluator(model, data=(data[1][0], data[1][1]), save_dir=model_dir)
        evaluator.generate_classification_report()
        evaluator.plot_confusion_matrix(save=True)

        logger = Logger(save_dir=model_dir)
        logger.set_history(history)
        logger.plot_training_history(save=True)

        # Save training time
        with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
            f.write(f"Training Time: {training_time:.2f} seconds")

        all_histories[current_mode] = {
            'history': history.history,
            'training_time': training_time
        }

    # Compare models if both were trained
    if mode == 'both':
        Logger.compare_histories(all_histories, save_dir=results_dir)

    # Image Upload and Prediction
    while True:
        test_option = input("\nDo you want to test the model with your own image? (yes/no): ").strip().lower()
        if test_option == 'yes':
            image_path = input("Enter the path to the image file: ").strip()
            if not os.path.isfile(image_path):
                print("Invalid file path. Please try again.")
                continue

            # Choose model for prediction
            model_choice = input("Select model for prediction (classical/quantum): ").strip().lower()
            if model_choice not in modes:
                print("Invalid model choice. Please try again.")
                continue

            # Load the saved model
            model = tf.keras.models.load_model(os.path.join(results_dir, model_choice, 'trained_model.h5'), compile=False)

            uploader = ImageUploader()
            processed_image = uploader.process_image(image_path)
            prediction = model.predict(processed_image)
            predicted_class = prediction.argmax()

            print(f"The model predicts this image as digit: {predicted_class}")
        else:
            break

if __name__ == '__main__':
    main()