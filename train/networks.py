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
