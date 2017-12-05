import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, dim_list, input_dim = 32 * 32 * 3, output_dim = 10):
        """Constructs a fully connected network with given hidden dimensions"""
        super(FC, self).__init__()
        self.fc_list = []
        for h_dim in dim_list:
            self.fc_list.append(nn.Linear(input_dim, h_dim))
            input_dim = h_dim

        self.classification = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        x = input

        for linear in self.fc_list:
            x = F.elu(linear(x))

        output = self.classification(x)

        return output
