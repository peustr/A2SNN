import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import LowRankMultivariateNormal

from resnet import resnet18


class GeneratorResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.rn = resnet18(False, zero_init_residual=True)

    def forward(self, x):
        x = self.rn(x)
        return x


class VanillaResNet18(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.gen = GeneratorResNet18()
        self.dim_reduction = nn.Linear(512, D)
        self.relu = nn.ReLU()
        self.proto = nn.Linear(D, C)

    def forward(self, x):
        x = self.gen(x)
        x = self.relu(self.dim_reduction(x))
        x = self.proto(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))


class SESNN_ResNet18(nn.Module):
    """ Trainable triangular matrix L, so Sigma=LL^T. """
    def __init__(self, D, C):
        super().__init__()
        self.gen = GeneratorResNet18()
        self.dim_reduction = nn.Linear(512, D)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.mu = nn.Parameter(torch.zeros(D), requires_grad=False)
        self.cov_factor = nn.Parameter(torch.rand(D))
        self.cov_diag = nn.Parameter(torch.rand(D))
        self.proto = nn.Linear(D, C)

    def forward(self, x):
        x = self.gen(x)
        x = self.relu(self.dim_reduction(x))
        self.dist = LowRankMultivariateNormal(self.mu, self.cov_factor, self.softplus(self.cov_diag))
        x_sample = self.dist.rsample()
        x = x + x_sample
        x = self.proto(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))


def model_factory(dataset, training_type, feature_dim):
    if dataset == 'cifar10':
        if training_type == 'vanilla':
            model = VanillaResNet18(feature_dim, 10)
        elif training_type == 'stochastic':
            model = SESNN_ResNet18(feature_dim, 10)
    else:
        raise NotImplementedError('Model for dataset {} not implemented.'.format(dataset))
    return model
