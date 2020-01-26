import numpy as np
import torch

"""
Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles, NIPS 2017
https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf
"""


class MLP(torch.nn.Module):
    def __init__(self, embedding_size, num_hidden, num_layer):
        super(MLP, self).__init__()
        self.num_layer = num_layer

        self.embedding_layer = torch.nn.Linear(embedding_size, num_hidden)
        self.mlps = torch.nn.ModuleList([
            torch.nn.Linear(num_hidden, num_hidden) for _ in range(num_layer)
        ])
        self.final_layer = torch.nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = self.embedding_layer(x)
        for i in range(self.num_layer):
            x = self.mlps[i](x)
        x = self.final_layer(x)
        return x

