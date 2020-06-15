import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from resnet import resnet18


class DataIndependentNoise(nn.Module):
    def __init__(self, H, device='cpu'):
        super().__init__()
        self.H = H
        self.device = device
        self.nop_input = torch.ones(H).to(self.device)
        self.fc_sigma = nn.Linear(H, H)

    def forward(self, x):
        sigma = self.fc_sigma(self.nop_input)
        dist = Normal(0., torch.sigmoid(sigma))
        x_sample = dist.rsample()
        return x + x_sample


class GeneratorResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.rn = resnet18(False, zero_init_residual=True)

    def forward(self, x):
        x = self.rn(x)
        return x


class SESNN_ResNet18(nn.Module):
    def __init__(self, C, device='cpu'):
        super().__init__()
        self.gen = GeneratorResNet18()
        self.noise = DataIndependentNoise(512, device=device)
        self.proto = nn.Linear(512, C)

    def forward(self, x):
        x = self.gen(x)
        x = self.noise(x)
        x = self.proto(x)
        return x


def model_factory(dataset, meta_train, device='cpu'):
    if dataset == 'cifar10':
        model = SESNN_ResNet18(10, device=device)
    else:
        raise NotImplementedError('Model for dataset {} not implemented.'.format(dataset))
    return model
