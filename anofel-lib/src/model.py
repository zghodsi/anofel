from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.optim import Adam
from torch.functional import F

from config import config, Batch

# 7.8K params
class SimpleLinear(torch.nn.Module):
    """Simple model for simple purposes"""

    def __init__(self, in_size: int, out_size: int):
        super().__init__()

        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)  # Make it flat
        x = self.linear(x)
        x = torch.relu(x)

        output = F.log_softmax(x, dim=1)

        return x

# 1.2M params
class SimpleCNN(torch.nn.Module):
    """Simple model for simple purposes"""

    def __init__(self, in_size: int, out_size: int):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return x


# 44.4K params
class OldLenet5(nn.Module):
    def __init__(self):
        super(OldLenet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x



# 61.7K params
# http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



class SimpleRNN(nn.Module):
    """
    Inputs:
        in_size = size of vocabulary
        out_size = num of categories
    """
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(in_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, inputs):
        #  inputs = inputs.view(-1, 28, 28)

        embedded = self.embedding(inputs)

        self.rnn.flatten_parameters()
        out, _ = self.rnn(embedded)

        out = self.out(out)
        out = out[:, -1, :]  # at last timestep

        out = F.log_softmax(out, dim=1)

        return out


# Currently used model to import from trainer
Model = Union[SimpleLinear, SimpleCNN, SimpleRNN]


