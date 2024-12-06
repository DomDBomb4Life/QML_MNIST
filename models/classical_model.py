# File: models/classical_model.py
import torch
import torch.nn as nn

def build_classical_model():
    """
    Builds a simple feedforward neural network for MNIST classification.
    
    Architecture:
    - Flatten input to (batch_size, 784)
    - Linear(784 -> 64)
    - ReLU
    - Linear(64 -> 10)
    - No softmax, as CrossEntropyLoss expects raw logits
    """
    class ClassicalModel(nn.Module):
        def __init__(self):
            super(ClassicalModel, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(784, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            # If x is (batch_size, 784), flatten will be no-op. Otherwise, ensures correct shape.
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            # No softmax here, CrossEntropyLoss expects logits
            return x

    return ClassicalModel()