import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from attacks.fgsm import fgsm
from attacks.pgd import pgd
from metrics import accuracy
from utils import normalize_cifar10, normalize_cifar100, normalize_generic


def train_vanilla(model, train_loader, test_loader, args, device='cpu'):
    optimizer = Adam(model.parameters(), lr=args['lr'])
    loss_func = nn.CrossEntropyLoss()
    if args['dataset'] == 'cifar10':
        norm_func = normalize_cifar10
    elif args['dataset'] == 'cifar100':
        norm_func = normalize_cifar100
    elif args['dataset'] == 'svhn':
        norm_func = normalize_generic
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
    # Comment out if using the w = w / w.norm() projection.
    # optimizer = Adam([
    #     {'params': model.base.parameters()},
    #     {'params': model.proto.parameters(), 'weight_decay': args['wd']}
    # ], lr=args['lr'])
    loss_func = nn.CrossEntropyLoss()
    if args['dataset'] == 'cifar10':
        norm_func = normalize_cifar10
    elif args['dataset'] == 'cifar100':
        norm_func = normalize_cifar100
    elif args['dataset'] == 'svhn':
        norm_func = normalize_generic
    elif args['dataset'] in ('mnist', 'fmnist'):
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
            if args['reg_type'] == 'wSw':
                if args['var_type'] == 'full_rank':
                    wSw = (model.proto.weight @ model.sigma @ model.proto.weight.T).diagonal().sum()
                elif args['var_type'] == 'diagonal':
                    wSw = (model.proto.weight @ model.sigma.diag() @ model.proto.weight.T).diagonal().sum()
                loss = loss_func(logits, target) - args['reg_weight'] * torch.log(wSw)
            elif args['reg_type'] == 'max_entropy':
                loss = loss_func(logits, target) - args['reg_weight'] * model.base.dist.entropy().mean()
            elif args['reg_type'] == 'wSw+max_entropy':
                if args['var_type'] == 'full_rank':
                    wSw = (model.proto.weight @ model.sigma @ model.proto.weight.T).diagonal().sum()
                elif args['var_type'] == 'diagonal':
                    wSw = (model.proto.weight @ model.sigma.diag() @ model.proto.weight.T).diagonal().sum()
                # In this case reg_weight needs to be an array with two items.
                loss = loss_func(logits, target)\
                    - args['reg_weight'][0] * model.base.dist.entropy().mean() - args['reg_weight'][1] * torch.log(wSw)
            else:
                raise NotImplementedError('Regularization "{}" not supported.'.format(args['reg_type']))
            loss.backward()
            optimizer.step()
            # For (w^T Sigma w) regularization, force w to have unit norm (so it doesn't explode).
            # Comment out if using weight decay.
            if (args['reg_type'] == 'wSw') or (args['reg_type'] == 'wSw+max_entropy'):
                with torch.no_grad():
                    model.proto.weight.data = model.proto.weight / model.proto.weight.norm()
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


def train_stochastic_adversarial(model, train_loader, test_loader, args, device='cpu'):
    optimizer = Adam(model.parameters(), lr=args['lr'])
    # Comment out if using the w = w / w.norm() projection.
    # optimizer = Adam([
    #     {'params': model.base.parameters()},
    #     {'params': model.proto.parameters(), 'weight_decay': args['wd']}
    # ], lr=args['lr'])
    loss_func = nn.CrossEntropyLoss()
    if args['dataset'] == 'cifar10':
        norm_func = normalize_cifar10
    elif args['dataset'] == 'cifar100':
        norm_func = normalize_cifar100
    elif args['dataset'] == 'svhn':
        norm_func = normalize_generic
    elif args['dataset'] in ('mnist', 'fmnist'):
        norm_func = None
    if args['attack'] == 'fgsm':
        attack_func = fgsm
    elif args['attack'] == 'pgd':
        attack_func = pgd
    epoch_margin = 10
    best_test_acc = -1.
    train_acc, test_acc = [], []
    sigma_hist = []
    for epoch in range(args['num_epochs'] + epoch_margin):
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            perturbed_data = attack_func(model, data, target, epsilon=args['epsilon']).to(device)
            model.train()
            if norm_func is not None:
                data = norm_func(data)
                perturbed_data = norm_func(perturbed_data)
            logits = model(data)
            adv_logits = model(perturbed_data)
            optimizer.zero_grad()
            if args['reg_type'] == 'wSw':
                wSw = (model.proto.weight @ model.sigma @ model.proto.weight.T).diagonal().sum()
                clean_loss = loss_func(logits, target)
                adv_loss = loss_func(adv_logits, target)
                loss = 0.5 * clean_loss + 0.5 * adv_loss - args['reg_weight'] * torch.log(wSw)
            elif args['reg_type'] == 'max_entropy':
                clean_loss = loss_func(logits, target)
                adv_loss = loss_func(adv_logits, target)
                loss = 0.5 * clean_loss + 0.5 * adv_loss - args['reg_weight'] * model.base.dist.entropy().mean()
            elif args['reg_type'] == 'wSw+max_entropy':
                wSw = (model.proto.weight @ model.sigma @ model.proto.weight.T).diagonal().sum()
                clean_loss = loss_func(logits, target)
                adv_loss = loss_func(adv_logits, target)
                # In this case reg_weight needs to be an array with two items.
                loss = 0.5 * clean_loss + 0.5 * adv_loss\
                    - args['reg_weight'][0] * model.base.dist.entropy().mean() - args['reg_weight'][1] * torch.log(wSw)
            else:
                raise NotImplementedError('Regularization "{}" not supported.'.format(args['reg_type']))
            loss.backward()
            optimizer.step()
            # For (w^T Sigma w) regularization, force w to have unit norm (so it doesn't explode).
            # Comment out if using weight decay.
            if (args['reg_type'] == 'wSw') or (args['reg_type'] == 'wSw+max_entropy'):
                with torch.no_grad():
                    model.proto.weight.data = model.proto.weight / model.proto.weight.norm()
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
