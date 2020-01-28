import numpy as np
import torch

"""
Probabilistic Backpropagation for ScalableLearning of Bayesian Neural Networks, ICML 2015
https://arxiv.org/pdf/1502.05336.pdf

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
            x = torch.nn.functional.relu(x)
        x = self.final_layer(x)
        return x


class GaussianMLP(torch.nn.Module):
    def __init__(self, embedding_size, num_hidden, num_layer):
        super(GaussianMLP, self).__init__()
        self.num_layer = num_layer

        self.embedding_layer = torch.nn.Linear(embedding_size, num_hidden)
        self.mlps = torch.nn.ModuleList([
            torch.nn.Linear(num_hidden, num_hidden) for _ in range(num_layer)
        ])
        self.mu_layer = torch.nn.Linear(num_hidden, 1)
        self.sigma_layer = torch.nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = self.embedding_layer(x)
        for i in range(self.num_layer):
            x = self.mlps[i](x)
            x = torch.nn.functional.relu(x)
        mu = self.mu_layer(x)
        sigma = self.sigma_layer(x)
        sigma = torch.nn.Softplus()(sigma) + 1e-6
        return mu, sigma
        # return torch.cat([mu, sigma], dim=1)

    def NLLLoss(self, mu, sigma, y):
        loss = torch.mean((torch.log(sigma) / 2) + (torch.pow((mu - y), 2) / (2 * sigma)))
        return loss
