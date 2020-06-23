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
        self.sigma = torch.rand(512, requires_grad=True, device=self.device)
        self.proto = nn.Linear(512, C)

    def forward(self, x):
        x = self.gen(x)
        sigma = f.softplus(self.sigma - 5.)
        self.dist = Normal(0., sigma)
        x_sample = self.dist.rsample()
        x = x + x_sample
        x = self.proto(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")
        torch.save(self.sigma, filename + "_sigma.pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))
        self.sigma = torch.load(filename + "_sigma.pt")


class MetaSESNN_ResNet18(nn.Module):
    """ Trainable sigma, b and reg. weight. """
    def __init__(self, C, device='cpu'):
        super().__init__()
        self.device = device
        self.gen = GeneratorResNet18()
        self.sigma = torch.rand(512, requires_grad=True, device=self.device)
        self.proto = nn.Linear(512, C)
        self.b = torch.rand(512, requires_grad=True, device=self.device)
        self.lambda2 = torch.rand(1, requires_grad=True, device=self.device)

    def forward(self, x):
        x = self.gen(x)
        sigma = f.softplus(self.sigma - 5.)
        self.dist = Normal(0., sigma)
        x_sample = self.dist.rsample()
        x = x + x_sample
        x = self.proto(x)
        return x

    def get_b(self):
        return f.softplus(self.b)

    def get_lambda2(self):
        return torch.sigmoid(self.lambda2)

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")
        torch.save(self.sigma, filename + "_sigma.pt")
        torch.save(self.b, filename + "_b.pt")
        torch.save(self.lambda2, filename + "_lambda2.pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))
        self.sigma = torch.load(filename + "_sigma.pt")
        self.b = torch.load(filename + "_b.pt")
        self.lambda2 = torch.load(filename + "_lambda2.pt")


def model_factory(dataset, meta_train, device='cpu'):
    if dataset == 'cifar10':
        if not meta_train:
            model = SESNN_ResNet18(10, device=device)
        else:
            model = MetaSESNN_ResNet18(10, device=device)
    else:
        raise NotImplementedError('Model for dataset {} not implemented.'.format(dataset))
    return model
