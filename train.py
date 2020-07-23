import math
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
    epoch_margin = 10
    best_test_acc = -1.
    train_acc, test_acc = [], []
    for epoch in range(args['num_epochs'] + epoch_margin):
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
        print('Epoch {:03}, Train acc: {:.3f}, Test acc: {:.3f}'.format(epoch + 1, train_acc[-1], test_acc[-1]))
        if epoch < args['num_epochs']:
            continue
        # Save model with best testing performance, after waiting for relative convergence.
        if test_acc[-1] > best_test_acc:
            best_test_acc = test_acc[-1]
            torch.save(model.state_dict(), os.path.join(args['output_path']['models'], 'ckpt_best.pt'))
    # Also save the training and testing curves.
    np.save(os.path.join(args['output_path']['stats'], 'train_acc.npy'), np.array(train_acc))
    np.save(os.path.join(args['output_path']['stats'], 'test_acc.npy'), np.array(test_acc))


def train_stochastic(model, train_loader, test_loader, args, device='cpu'):
    optimizer = Adam(model.parameters(), lr=args['lr'])
    loss_func = nn.CrossEntropyLoss()
    if args['dataset'] == 'cifar10':
        norm_func = normalize_cifar10
    elif args['dataset'] == 'mnist':
        norm_func = None
    epoch_margin = 10
    best_test_acc = -1.
    train_acc, test_acc = [], []
    sigma_hist = []
    for epoch in range(args['num_epochs'] + epoch_margin):
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            model.train()
            if norm_func is not None:
                data = norm_func(data)
            logits = model(data)
            optimizer.zero_grad()
            # (w^T Sigma w) regularization.
            if args['reg_type'] == 'wSw':
                # Force w to have unit norm (so it doesn't explode).
                with torch.no_grad():
                    model.proto.weight = model.proto.weight / model.proto.weight.norm()
                omega = (model.proto.weight @ model.sigma @ model.proto.weight.T).diagonal().sum()
                loss = loss_func(logits, target) + args['reg_weight'] * omega
            elif args['reg_type'] == 'max_ent':
                threshold = math.log(args['var_threshold']) + (1 + math.log(2 * math.pi)) / 2
                entropy_loss = torch.relu(threshold - model.base.dist.entropy()).mean()
                loss = loss_func(logits, target) + args['reg_weight'] * entropy_loss
            else:
                raise NotImplementedError('Regularization "{}" not supported.'.format(args['reg_type']))
            loss.backward()
            optimizer.step()
        train_acc.append(accuracy(model, train_loader, device=device, norm=norm_func))
        test_acc.append(accuracy(model, test_loader, device=device, norm=norm_func))
        sigma_hist.append(model.sigma.detach().cpu().numpy())
        print('Epoch {:03}, Train acc: {:.3f}, Test acc: {:.3f}'.format(epoch + 1, train_acc[-1], test_acc[-1]))
        if epoch < args['num_epochs']:
            continue
        # Save model with best testing performance.
        if test_acc[-1] > best_test_acc:
            best_test_acc = test_acc[-1]
            model.save(os.path.join(args['output_path']['models'], 'ckpt_best'))
    # Also save the training and testing curves.
    np.save(os.path.join(args['output_path']['stats'], 'train_acc.npy'), np.array(train_acc))
    np.save(os.path.join(args['output_path']['stats'], 'test_acc.npy'), np.array(test_acc))
    np.save(os.path.join(args['output_path']['stats'], 'sigma_hist.npy'), np.array(sigma_hist))
