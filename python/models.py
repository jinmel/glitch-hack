from torch import nn
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(16, 160)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(160, 1)
        self.sigmoid = nn.Sigmoid()

    def std_scale(self, x):
        m = x.mean(0, keepdim=True)
        s = x.std(0, unbiased=False, keepdim=True)
        x = x - m
        x = x / s
        return x

    def forward(self, x):
        # StandardScaler.
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x
