# import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from metrics import accuracy
from utils import normalize_cifar10


def train_vanilla(model, train_loader, test_loader, args, device='cpu'):
    optimizer = Adam(model.parameters(), lr=args['lr'])
    loss_func = nn.CrossEntropyLoss()
    if args['dataset'] == 'cifar10':
        norm_func = normalize_cifar10
    else:
        norm_func = None
    best_test_acc = -1.
    train_acc, test_acc = [], []
    for epoch in range(args['num_epochs']):
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            model.train()
            if norm_func is not None:
                data = norm_func(data)
            logits = model(data)
            optimizer.zero_grad()
            loss = loss_func(logits, target)
            loss.backward()
            optimizer.step()
        train_acc.append(accuracy(model, train_loader, device=device, norm=norm_func))
        test_acc.append(accuracy(model, test_loader, device=device, norm=norm_func))
        # Checkpoint current model.
        torch.save(model.state_dict(), os.path.join(args['output_path']['models'], 'ckpt.pt'))
        # Save model with best testing performance.
        if test_acc[-1] > best_test_acc:
            best_test_acc = test_acc[-1]
            torch.save(model.state_dict(), os.path.join(args['output_path']['models'], 'ckpt_best.pt'))
        print('Epoch {:03}, Train acc: {:.3f}, Test acc: {:.3f}'.format(epoch + 1, train_acc[-1], test_acc[-1]))
    # Also save the training and testing curves.
    np.save(os.path.join(args['output_path']['stats'], 'train_acc.npy'), np.array(train_acc))
    np.save(os.path.join(args['output_path']['stats'], 'test_acc.npy'), np.array(test_acc))


def train_stochastic(model, train_loader, test_loader, args, device='cpu'):
    optimizer = Adam(model.parameters(), lr=args['lr'])
    loss_func = nn.CrossEntropyLoss()
    if args['dataset'] == 'cifar10':
        norm_func = normalize_cifar10
    else:
        norm_func = None
    best_test_acc = -1.
    train_acc, test_acc = [], []
    sigma_hist = []
    for epoch in range(args['num_epochs']):
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            model.train()
            if norm_func is not None:
                data = norm_func(data)
            logits = model(data)
            optimizer.zero_grad()
            # Max entropy regularization.
            # threshold = math.log(args['var_threshold']) + (1 + math.log(2 * math.pi)) / 2
            # max_entropy_reg = torch.relu(threshold - model.dist.entropy()).mean()
            # loss = loss_func(logits, target) + args['reg_weight'] * max_entropy_reg
            # w^T Sigma w regularization.
            omega = torch.sum(model.proto.weight.sum(dim=0).T * model.sigma * model.proto.weight.sum(dim=0))
            loss = loss_func(logits, target) + args['reg_weight'] * omega
            loss.backward()
            optimizer.step()
        train_acc.append(accuracy(model, train_loader, device=device, norm=norm_func))
        test_acc.append(accuracy(model, test_loader, device=device, norm=norm_func))
        sigma_hist.append(model.sigma.detach().cpu().numpy())
        # Checkpoint current model.
        model.save(os.path.join(args['output_path']['models'], 'ckpt'))
        # Save model with best testing performance.
        if test_acc[-1] > best_test_acc:
            best_test_acc = test_acc[-1]
            model.save(os.path.join(args['output_path']['models'], 'ckpt_best'))
        print('Epoch {:03}, Train acc: {:.3f}, Test acc: {:.3f}'.format(epoch + 1, train_acc[-1], test_acc[-1]))
    # Also save the training and testing curves.
    np.save(os.path.join(args['output_path']['stats'], 'train_acc.npy'), np.array(train_acc))
    np.save(os.path.join(args['output_path']['stats'], 'test_acc.npy'), np.array(test_acc))
    np.save(os.path.join(args['output_path']['stats'], 'sigma_hist.npy'), np.array(sigma_hist))
