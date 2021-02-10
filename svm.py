import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from attacks.pgd import pgd


class SVM_Isotropic(nn.Module):
    def __init__(self, D):
        super(SVM_Isotropic, self).__init__()
        self.dimr = nn.Linear(D, 32)
        self.fc = nn.Linear(32, 1)
        self.sigma = nn.Parameter(torch.rand(32))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.dimr(x))
        self.dist = Normal(0., F.softplus(self.sigma))
        x_sample = self.dist.rsample()
        x = x + x_sample
        x = self.fc(x)
        return x


class SVM_Anisotropic(nn.Module):
    def __init__(self, D):
        super(SVM_Anisotropic, self).__init__()
        self.dimr = nn.Linear(D, 32)
        self.fc = nn.Linear(32, 1)
        self.mu = nn.Parameter(torch.zeros(32), requires_grad=False)
        self.L = nn.Parameter(torch.rand(32, 32))

    @property
    def sigma(self):
        return self.L @ self.L.T

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.dimr(x))
        self.dist = MultivariateNormal(self.mu, scale_tril=self.L)
        x_sample = self.dist.rsample()
        x = x + x_sample
        x = self.fc(x)
        return x


class CustomMNIST(Dataset):
    def __init__(self, train, transform):
        self.dataset = datasets.MNIST('./data', train=train, transform=transform)
        self.global_index = 0

    def __getitem__(self, idx):
        try:
            d, t = self.dataset[self.global_index]
        except IndexError:
            self.global_index = 0
            d, t = self.dataset[self.global_index]
        while t not in [0, 1]:
            self.global_index += 1
            try:
                d, t = self.dataset[self.global_index]
            except IndexError:
                self.global_index = 0
                d, t = self.dataset[self.global_index]
        if t == 0:
            t = -1
        elif t == 1:
            t = 1
        self.global_index += 1
        return d, t

    def __len__(self):
        return len(self.dataset)


def export():
    device = 'cuda:0'
    model = SVM_Isotropic(28 * 28)
    # model = SVM_Ansotropic(28 * 28)
    model.to(device)
    model.load_state_dict(torch.load("q_i.pt"))
    torch.save(model.sigma.detach(), 'sigma_i.pt')
    torch.save(model.fc.weight.data, 'svm_weights_i.pt')
    # model.load_state_dict(torch.load("q_a.pt"))
    # torch.save(model.sigma.detach(), 'sigma_a.pt')
    # torch.save(model.fc.weight.data, 'svm_weights_a.pt')


def main():
    device = 'cuda:0'
    bs = 128
    model = SVM_Isotropic(28 * 28)
    # model = SVM_Anisotropic(28 * 28)
    model.to(device)
    # model.load_state_dict(torch.load("q_i.pt"))
    # model.load_state_dict(torch.load("q_a.pt"))
    optimizer = Adam(model.parameters())
    tr = transforms.Compose([transforms.ToTensor()])
    train_set = CustomMNIST(train=True, transform=tr)
    test_set = CustomMNIST(train=False, transform=tr)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, drop_last=True)
    loss_threshold = 100.
    print("training...")
    losses = []
    for epoch in range(100):
        break
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            logit = model(data)
            optimizer.zero_grad()
            wca = (model.fc.weight @ model.sigma.diag() @ model.fc.weight.T).diagonal().sum()
            # wca = (model.fc.weight @ model.sigma @ model.fc.weight.T).diagonal().sum()
            z = torch.zeros(bs).to(device)
            o = torch.ones(bs).to(device)
            hinge_loss = torch.max(z, o - target * logit.squeeze()).mean()
            loss = hinge_loss - 1e-5 * torch.log(wca)
            loss.backward()
            optimizer.step()
            # with torch.no_grad():
            #     model.L.data = model.L.data.tril()
            losses.append(hinge_loss.item())
        mean_loss = np.mean(losses)
        print('epoch: {}, mean loss: {}'.format(epoch + 1, mean_loss))
        if mean_loss < loss_threshold:
            loss_threshold = mean_loss
            torch.save(model.state_dict(), "q_i.pt")
            print('saving model...')
    print("testing...")
    accs = []
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        perturbed_data = pgd(model, data, target, epsilon=4, k=10)
        logit = model(perturbed_data)
        logit = logit.squeeze()
        target = target.squeeze()
        logit[logit < 0] = -1
        logit[logit > 0] = 1
        acc = (logit == target).sum().item() / len(logit)
        accs.append(acc)
    print(np.mean(acc))


if __name__ == '__main__':
    main()
    # export()
