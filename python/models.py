from torch import nn
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(16, 320)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(320, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # StandardScaler.
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x
