from mpl_toolkits.mplot3d import Axes3D  # noqa

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import torch  # noqa
import torch.nn.functional as f

from data_loaders import get_data_loader
from models import model_factory


def main():
    # m0_path = None  # Replace with path to model
    # m2_path = None
    m4_path = None

    # model = model_factory('cifar10', 'vanilla', None, 32, 10)
    # model.load(m0_path)
    model = model_factory('cifar10', 'stochastic', 'anisotropic', 32, 10)
    model.load(m4_path)
    # model.base.disable_noise = True
    model.eval()

    dl = get_data_loader('cifar10', 1, train=False, shuffle=False, drop_last=False)

    skip = 0
    for data, target in dl:
        u = data
        t = target
        if skip == 0:
            break
        else:
            skip += 1

    X, Y, Z = [], [], []

    u.requires_grad = True
    if u.grad is not None:
        u.grad.data.zero_()
    logits = model(u)
    # print(logits.argmax(), t)
    loss = f.cross_entropy(logits, t)
    loss.backward()
    u.requires_grad = False
    v1 = u.grad.data.reshape(1 * 3 * 32 * 32)
    # v1 = torch.FloatTensor(1 * 3 * 32 * 32).uniform_(-1, 1)
    v1 = v1 / v1.norm()
    v2 = v1.clone()
    v2[-1] = -(v1[:-1] @ v1[:-1]) / v1[-1]
    v2 = v2 / v2.norm()
    # print(v1 @ v2)

    num_iter = 50
    u_ = u.reshape(1 * 3 * 32 * 32)
    epsilons = np.linspace(-0.1, 0.1, 50)
    for e1 in epsilons:
        for e2 in epsilons:
            losses = []
            for i in range(num_iter):
                u_prime = (u_ + e1 * v1 + e2 * v2).reshape(1, 3, 32, 32)
                logits = model(u_prime)
                loss = f.cross_entropy(logits, t)
                losses.append(loss.item())
            X.append(e1)
            Y.append(e2)
            Z.append(np.mean(losses))

    # print('X', X)
    # print('Y', Y)
    # print('Z', Z)

    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Grad')
    ax.set_ylabel('Grad orth.')
    ax.set_zlabel('Loss')
    ax.set_title('WCA, anisotropic, stochastic')

    surf = ax.plot_trisurf(X, Y, Z, cmap=cm.Blues, linewidth=0, antialiased=False)  # noqa

    plt.savefig('m4.png')
    print('Done!')


if __name__ == '__main__':
    main()
