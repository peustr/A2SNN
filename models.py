import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.multivariate_normal import MultivariateNormal

from resnet import resnet18


class Generator(nn.Module):
    def __init__(self, D):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 32, D)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 32)
        x = f.relu(self.fc1(x))
        return x


class GeneratorResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.rn = resnet18(False, zero_init_residual=True)

    def forward(self, x):
        x = self.rn(x)
        return x


class VanillaNet(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.gen = Generator()
        self.proto = nn.Linear(D, C)

    def forward(self, x):
        x = self.gen(x)
        x = self.proto(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))


class VanillaResNet18(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.gen = GeneratorResNet18()
        self.fc1 = nn.Linear(512, D)
        self.proto = nn.Linear(D, C)

    def forward(self, x):
        x = self.gen(x)
        x = f.relu(self.fc1(x))
        x = self.proto(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))


class StochasticBase(nn.Module):
    """ Trainable triangular matrix L, so Sigma=LL^T. """
    def __init__(self, D):
        super().__init__()
        self.gen = Generator(D)
        self.mu = nn.Parameter(torch.zeros(D), requires_grad=False)
        self.L = nn.Parameter(torch.rand(D, D))

    def forward(self, x):
        x = self.gen(x)
        self.dist = MultivariateNormal(self.mu, scale_tril=self.L)
        x_sample = self.dist.rsample()
        x = x + x_sample
        return x


class ResNet18_StochasticBase(nn.Module):
    """ Trainable triangular matrix L, so Sigma=LL^T. """
    def __init__(self, D):
        super().__init__()
        self.gen = GeneratorResNet18()
        self.fc1 = nn.Linear(512, D)
        self.mu = nn.Parameter(torch.zeros(D), requires_grad=False)
        self.L = nn.Parameter(torch.rand(D, D))

    def forward(self, x):
        x = self.gen(x)
        x = f.relu(self.fc1(x))
        self.dist = MultivariateNormal(self.mu, scale_tril=self.L)
        x_sample = self.dist.rsample()
        x = x + x_sample
        return x


class SESNN_CNN(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.base = StochasticBase(D)
        self.proto = nn.Linear(D, C)

    @property
    def sigma(self):
        return self.base.L @ self.base.L.T

    def forward(self, x):
        x = self.base(x)
        x = self.proto(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))


class SESNN_ResNet18(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.base = ResNet18_StochasticBase(D)
        self.proto = nn.Linear(D, C)

    @property
    def sigma(self):
        return self.base.L @ self.base.L.T

    def forward(self, x):
        x = self.base(x)
        x = self.proto(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))


def model_factory(dataset, training_type, feature_dim):
    if dataset == 'mnist':
        if training_type == 'vanilla':
            model = VanillaNet(feature_dim, 10)
        elif training_type == 'stochastic':
            model = SESNN_CNN(feature_dim, 10)
    elif dataset == 'cifar10':
        if training_type == 'vanilla':
            model = VanillaResNet18(feature_dim, 10)
        elif training_type == 'stochastic':
            model = SESNN_ResNet18(feature_dim, 10)
    else:
        raise NotImplementedError('Model for dataset {} not implemented.'.format(dataset))
    return model
