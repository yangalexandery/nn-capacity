import torch
import torch.nn as nn
import torch.nn.functional as F


def FC(dim_list, input_dim = 32 * 32 * 3, output_dim = 10):
    """Constructs a fully connected network with given hidden dimensions"""
    modules = []

    for h_dim in dim_list:
        modules.append(nn.Linear(input_dim, h_dim))
        modules.append(nn.ELU())
        input_dim = h_dim

    modules.append(nn.Linear(input_dim, output_dim))

    return nn.Sequential(*modules)


class ConvSimple(nn.Module):
    def __init__(self):
        super(ConvSimple, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
