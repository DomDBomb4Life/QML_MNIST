# File: utils/data_loader.py
import numpy as np
from torchvision import datasets, transforms

class DataLoader:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def load_data(self):
        # Load MNIST using torchvision
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
        
        # Convert to NumPy arrays
        x_train = train_dataset.data.numpy().astype('float32') / 255.0
        y_train = train_dataset.targets.numpy()
        x_test = test_dataset.data.numpy().astype('float32') / 255.0
        y_test = test_dataset.targets.numpy()

        # Flatten the images to (num_samples, 784)
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)

        # Labels remain integers (no one-hot), suitable for CrossEntropyLoss
        return (x_train, y_train), (x_test, y_test)
