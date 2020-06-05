import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal

from resnet import resnet18


class RegTerm(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.nop_input = torch.ones(1).to(self.device)
        self.reg_term = nn.Linear(1, 1)

    def forward(self, x):
        return x

    def item(self):
        return self.reg_term(self.nop_input)


class DataIndependentNoise(nn.Module):
    def __init__(self, H, b=5., device='cpu'):
        super().__init__()
        self.H = H
        self.b = b
        self.device = device
        self.nop_input = torch.ones(H).to(self.device)
        self.fc_mu = nn.Linear(H, H)
        self.fc_sigma = nn.Linear(H, H)

    def forward(self, x):
        mu = self.fc_mu(self.nop_input)
        var = self.fc_var(self.nop_input)
        dist = Normal(mu, f.softplus(var - self.b))
        x_sample = dist.rsample()
        return x + x_sample


class DataIndependentMetaNoise(nn.Module):
    """ Trainable b. """
    def __init__(self, H, device='cpu'):
        super().__init__()
        self.H = H
        self.device = device
        self.nop_input = torch.ones(H).to(self.device)
        self.fc_mu = nn.Linear(H, H)
        self.fc_sigma = nn.Linear(H, H)
        self.fc_b = nn.Linear(H, H)

    def forward(self, x):
        mu = self.fc_mu(self.nop_input)
        var = self.fc_var(self.nop_input)
        b = self.fc_b(self.nop_input)
        dist = Normal(mu, f.softplus(var - b))
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


class MetaSESNN_ResNet18(nn.Module):
    """ Trainable b and reg. term. """
    def __init__(self, C, device='cpu'):
        super().__init__()
        self.gen = GeneratorResNet18()
        self.noise = DataIndependentMetaNoise(512, device=device)
       	self.proto = nn.Linear(512, C)
        self.reg_term = RegTerm()

    def	forward(self, x):
       	x = self.gen(x)
        x = self.noise(x)
        x = self.proto(x)
        return x

    def get_reg_term(self):
        return self.reg_term.item()
