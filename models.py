import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal

from resnet import resnet18


class GeneratorResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.rn = resnet18(False, zero_init_residual=True)

    def forward(self, x):
        x = self.rn(x)
        return x


class SESNN_ResNet18(nn.Module):
    """ Trainable sigma. """
    def __init__(self, C, device='cpu'):
        super().__init__()
        self.device = device
        self.gen = GeneratorResNet18()
        self.sigma = nn.Parameter(torch.rand(512))
        self.proto = nn.Linear(512, C)

    def forward(self, x):
        x = self.gen(x)
        self.dist = Normal(0., f.softplus(self.sigma))
        x_sample = self.dist.rsample()
        x = x + x_sample
        x = self.proto(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))


def model_factory(dataset, meta_train, device='cpu'):
    if dataset == 'cifar10':
        model = SESNN_ResNet18(10, device=device)
    else:
        raise NotImplementedError('Model for dataset {} not implemented.'.format(dataset))
    return model
