import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from metrics import accuracy
from test import test_attack
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
    # optimizer = Adam(model.parameters(), lr=args['lr'])
    optimizer = Adam([
        {'params': model.base.parameters()},
        {'params': model.proto.parameters(), 'weight_decay': 0.01}
    ], lr=args['lr'])
    loss_func = nn.CrossEntropyLoss()
    if args['dataset'] == 'cifar10':
        norm_func = normalize_cifar10
        eps_values = [0. / 255, 1. / 255, 2. / 255, 4. / 255, 8. / 255, 16. / 255, 32. / 255, 64. / 255, 128. / 255]
    elif args['dataset'] == 'mnist':
        norm_func = None
        eps_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    best_test_acc, best_test_acc_atk = -1., -1.
    train_acc, test_acc, test_acc_atk = [], [], []
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
            # (w^T Sigma w) regularization.
            if args['reg_type'] == 'wSw':
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
        test_acc_atk.append(test_attack(model, test_loader, 'FGSM', eps_values, args, device).mean().item())
        sigma_hist.append(model.sigma.detach().cpu().numpy())
        # Checkpoint current model.
        model.save(os.path.join(args['output_path']['models'], 'ckpt'))
        # Save model with best testing performance.
        if test_acc[-1] > best_test_acc:
            best_test_acc = test_acc[-1]
            model.save(os.path.join(args['output_path']['models'], 'ckpt_best'))
        if test_acc_atk[-1] > best_test_acc_atk:
            best_test_acc_atk = test_acc_atk[-1]
            model.save(os.path.join(args['output_path']['models'], 'ckpt_robust'))
        print('Epoch {:03}, Train acc: {:.3f}, Test acc: {:.3f}, Test adv. acc: {:.3f}'.format(
            epoch + 1, train_acc[-1], test_acc[-1], test_acc_atk[-1]))
    # Also save the training and testing curves.
    np.save(os.path.join(args['output_path']['stats'], 'train_acc.npy'), np.array(train_acc))
    np.save(os.path.join(args['output_path']['stats'], 'test_acc.npy'), np.array(test_acc))
    np.save(os.path.join(args['output_path']['stats'], 'test_acc_atk.npy'), np.array(test_acc_atk))
    np.save(os.path.join(args['output_path']['stats'], 'sigma_hist.npy'), np.array(sigma_hist))
