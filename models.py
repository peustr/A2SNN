import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from resnet import resnet18


class Generator(nn.Module):
    """ LeNets++ architecture from: "A Discriminative Feature Learning Approach for Deep Face Recognition"
        https://ydwen.github.io/papers/WenECCV16.pdf
    """
    def __init__(self, D):
        super(Generator, self).__init__()
        self.conv1 = self._make_conv_layer(1, 32, 5, 1, 2)
        self.conv2 = self._make_conv_layer(32, 32, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv3 = self._make_conv_layer(32, 64, 5, 1, 2)
        self.conv4 = self._make_conv_layer(64, 64, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv5 = self._make_conv_layer(64, 128, 5, 1, 2)
        self.conv6 = self._make_conv_layer(128, 128, 5, 1, 2)
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0)
        self.fc1 = nn.Linear(1152, D)

    def _make_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        prelu = nn.PReLU()
        return nn.Sequential(conv, prelu)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
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
        self.gen = Generator(D)
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


class StochasticBaseDiagonal(nn.Module):
    """ Zero mean, trainable variance. """
    def __init__(self, D):
        super().__init__()
        self.gen = Generator(D)
        self.sigma = nn.Parameter(torch.rand(D))

    def forward(self, x):
        x = self.gen(x)
        self.dist = Normal(0., f.softplus(self.sigma))
        x_sample = self.dist.rsample()
        x = x + x_sample
        return x


class StochasticBaseMultivariate(nn.Module):
    """ Trainable triangular matrix L, so Sigma=LL^T. """
    def __init__(self, D):
        super().__init__()
        self.gen = Generator(D)
        self.mu = nn.Parameter(torch.zeros(D), requires_grad=False)
        self.L = nn.Parameter(torch.rand(D, D))

    @property
    def sigma(self):
        return self.L.tril() @ self.L.tril().T

    def forward(self, x):
        x = self.gen(x)
        self.dist = MultivariateNormal(self.mu, scale_tril=self.L.tril())
        x_sample = self.dist.rsample()
        x = x + x_sample
        return x


class ResNet18_StochasticBaseDiagonal(nn.Module):
    """ Zero mean, trainable variance. """
    def __init__(self, D):
        super().__init__()
        self.gen = GeneratorResNet18()
        self.fc1 = nn.Linear(512, D)
        self.sigma = nn.Parameter(torch.rand(D))

    def forward(self, x):
        x = self.gen(x)
        x = f.relu(self.fc1(x))
        self.dist = Normal(0., f.softplus(self.sigma))
        x_sample = self.dist.rsample()
        x = x + x_sample
        return x


class ResNet18_StochasticBaseMultivariate(nn.Module):
    """ Trainable triangular matrix L, so Sigma=LL^T. """
    def __init__(self, D):
        super().__init__()
        self.gen = GeneratorResNet18()
        self.fc1 = nn.Linear(512, D)
        self.mu = nn.Parameter(torch.zeros(D), requires_grad=False)
        self.L = nn.Parameter(torch.rand(D, D))

    @property
    def sigma(self):
        return self.L.tril() @ self.L.tril().T

    def forward(self, x):
        x = self.gen(x)
        x = f.relu(self.fc1(x))
        self.dist = MultivariateNormal(self.mu, scale_tril=self.L.tril())
        x_sample = self.dist.rsample()
        x = x + x_sample
        return x


class SESNN_CNN(nn.Module):
    def __init__(self, D, C, variance_type):
        super().__init__()
        if variance_type == 'full_rank':
            self.base = StochasticBaseMultivariate(D)
        else:
            self.base = StochasticBaseDiagonal(D)
        self.proto = nn.Linear(D, C)

    @property
    def sigma(self):
        return self.base.sigma

    def forward(self, x):
        x = self.base(x)
        x = self.proto(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))


class SESNN_ResNet18(nn.Module):
    def __init__(self, D, C, variance_type):
        super().__init__()
        if variance_type == 'full_rank':
            self.base = ResNet18_StochasticBaseMultivariate(D)
        else:
            self.base = ResNet18_StochasticBaseDiagonal(D)
        self.proto = nn.Linear(D, C)

    @property
    def sigma(self):
        return self.base.sigma

    def forward(self, x):
        x = self.base(x)
        x = self.proto(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))


def model_factory(dataset, training_type, variance_type, feature_dim):
    if variance_type is not None and variance_type not in ('diagonal', 'full_rank'):
        raise NotImplementedError('Only "diagonal" and "full_rank" variance types supported.')
    if dataset == 'mnist':
        if training_type == 'vanilla':
            model = VanillaNet(feature_dim, 10)
        elif training_type == 'stochastic':
            model = SESNN_CNN(feature_dim, 10, variance_type)
    elif dataset == 'fmnist':
        if training_type == 'vanilla':
            model = VanillaNet(feature_dim, 10)
        elif training_type == 'stochastic':
            model = SESNN_CNN(feature_dim, 10, variance_type)
    elif dataset == 'cifar10':
        if training_type == 'vanilla':
            model = VanillaResNet18(feature_dim, 10)
        elif training_type == 'stochastic':
            model = SESNN_ResNet18(feature_dim, 10, variance_type)
    elif dataset == 'cifar100':
        if training_type == 'vanilla':
            model = VanillaResNet18(feature_dim, 100)
        elif training_type == 'stochastic':
            model = SESNN_ResNet18(feature_dim, 100, variance_type)
    elif dataset == 'svhn':
        if training_type == 'vanilla':
            model = VanillaResNet18(feature_dim, 10)
        elif training_type == 'stochastic':
            model = SESNN_ResNet18(feature_dim, 10, variance_type)
    else:
        raise NotImplementedError('Model for dataset {} not implemented.'.format(dataset))
    return model
