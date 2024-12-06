# File: models/classical_model.py
import torch
import torch.nn as nn

def build_classical_model():
    """
    Builds a classical feedforward network for MNIST:
    Input: (batch_size, 784)
    fc1: (784->64), ReLU
    fc2: (64->10)
    """
    class ClassicalModel(nn.Module):
        def __init__(self):
            super(ClassicalModel, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(784, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 10)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.softmax(x)  # Convert to probabilities
            return x

    return ClassicalModel()