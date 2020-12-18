import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from resnet import resnet18


class Generator(nn.Module):
    """ LeNets++ architecture from: "A Discriminative Feature Learning Approach for Deep Face Recognition"
        The variant is used by PCL, i.e. no max pooling and no padding.
    """
    def __init__(self, D):
        super(Generator, self).__init__()
        self.conv1 = self._make_conv_layer(1, 32, 5)
        self.conv2 = self._make_conv_layer(32, 32, 5)
        self.conv3 = self._make_conv_layer(32, 64, 5)
        self.conv4 = self._make_conv_layer(64, 64, 5)
        self.conv5 = self._make_conv_layer(64, 128, 5)
        self.conv6 = self._make_conv_layer(128, 128, 5)
        self.fc1 = nn.Linear(2048, D)

    def _make_conv_layer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size), nn.PReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
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
    """ Trainable lower triangular matrix L, so Sigma=LL^T. """
    def __init__(self, D):
        super().__init__()
        self.gen = Generator(D)
        self.mu = nn.Parameter(torch.zeros(D), requires_grad=False)
        self.L = nn.Parameter((torch.eye(D) + torch.rand(D, D)).tril())

    @property
    def sigma(self):
        return self.L @ self.L.T

    def forward(self, x):
        x = self.gen(x)
        self.dist = MultivariateNormal(self.mu, scale_tril=self.L)
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
    """ Trainable lower triangular matrix L, so Sigma=LL^T. """
    def __init__(self, D):
        super().__init__()
        self.gen = GeneratorResNet18()
        self.fc1 = nn.Linear(512, D)
        self.mu = nn.Parameter(torch.zeros(D), requires_grad=False)
        self.L = nn.Parameter((torch.eye(D) + torch.rand(D, D)).tril())

    @property
    def sigma(self):
        return self.L @ self.L.T

    def forward(self, x):
        x = self.gen(x)
        x = f.relu(self.fc1(x))
        self.dist = MultivariateNormal(self.mu, scale_tril=self.L)
        x_sample = self.dist.rsample()
        x = x + x_sample
        return x


class A2SNN_CNN(nn.Module):
    def __init__(self, D, C, variance_type):
        super().__init__()
        if variance_type == 'isotropic':
            self.base = StochasticBaseDiagonal(D)
        elif variance_type == 'anisotropic':
            self.base = StochasticBaseMultivariate(D)
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


class A2SNN_ResNet18(nn.Module):
    def __init__(self, D, C, variance_type):
        super().__init__()
        if variance_type == 'isotropic':
            self.base = ResNet18_StochasticBaseDiagonal(D)
        elif variance_type == 'anisotropic':
            self.base = ResNet18_StochasticBaseMultivariate(D)
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


def model_factory(dataset, training_type, variance_type, feature_dim, num_classes):
    if variance_type is not None and variance_type not in ('isotropic', 'anisotropic'):
        raise NotImplementedError('Only "isotropic" and "anisotropic" variance types supported.')
    if dataset == 'mnist':
        if training_type == 'vanilla':
            model = VanillaNet(feature_dim, num_classes)
        elif training_type in ('stochastic', 'stochastic+adversarial'):
            model = A2SNN_CNN(feature_dim, num_classes, variance_type)
    elif dataset == 'fmnist':
        if training_type == 'vanilla':
            model = VanillaNet(feature_dim, num_classes)
        elif training_type in ('stochastic', 'stochastic+adversarial'):
            model = A2SNN_CNN(feature_dim, num_classes, variance_type)
    elif dataset == 'cifar10':
        if training_type == 'vanilla':
            model = VanillaResNet18(feature_dim, num_classes)
        elif training_type in ('stochastic', 'stochastic+adversarial'):
            model = A2SNN_ResNet18(feature_dim, num_classes, variance_type)
    elif dataset == 'cifar100':
        if training_type == 'vanilla':
            model = VanillaResNet18(feature_dim, num_classes)
        elif training_type in ('stochastic', 'stochastic+adversarial'):
            model = A2SNN_ResNet18(feature_dim, num_classes, variance_type)
    elif dataset == 'svhn':
        if training_type == 'vanilla':
            model = VanillaResNet18(feature_dim, num_classes)
        elif training_type in ('stochastic', 'stochastic+adversarial'):
            model = A2SNN_ResNet18(feature_dim, num_classes, variance_type)
    else:
        raise NotImplementedError('Model for dataset {} not implemented.'.format(dataset))
    return model
