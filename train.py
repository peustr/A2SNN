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
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    loss_func = nn.CrossEntropyLoss()
    if args['dataset'] == 'cifar10':
        norm_func = normalize_cifar10
    elif args['dataset'] == 'cifar100':
        norm_func = normalize_cifar100
    elif args['dataset'] == 'svhn':
        norm_func = normalize_generic
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
        print('Epoch {:03}, Train acc: {:.3f}, Test acc: {:.3f}'.format(epoch + 1, train_acc[-1], test_acc[-1]))
        if test_acc[-1] > best_test_acc:
            best_test_acc = test_acc[-1]
            torch.save(model.state_dict(), os.path.join(args['output_path']['models'], 'ckpt_best.pt'))
    np.save(os.path.join(args['output_path']['stats'], 'train_acc.npy'), np.array(train_acc))
    np.save(os.path.join(args['output_path']['stats'], 'test_acc.npy'), np.array(test_acc))


def train_stochastic(model, train_loader, test_loader, args, device='cpu'):
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    loss_func = nn.CrossEntropyLoss()
    if args['dataset'] == 'cifar10':
        norm_func = normalize_cifar10
    elif args['dataset'] == 'cifar100':
        norm_func = normalize_cifar100
    elif args['dataset'] == 'svhn':
        norm_func = normalize_generic
    elif args['dataset'] in ('mnist', 'fmnist'):
        norm_func = None
    epsilon = 1e-3
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
            if args['reg_type'] == 'wca':
                if args['var_type'] == 'isotropic':
                    wca = (model.proto.weight @ model.sigma.diag() @ model.proto.weight.T).diagonal().sum()
                elif args['var_type'] == 'anisotropic':
                    wca = (model.proto.weight @ model.sigma @ model.proto.weight.T).diagonal().sum()
                loss = loss_func(logits, target) - wca
            elif args['reg_type'] == 'max_entropy':
                me = torch.log(model.base.dist.entropy().mean())
                loss = loss_func(logits, target) - me
            elif args['reg_type'] == 'wca+max_entropy':
                if args['var_type'] == 'isotropic':
                    wca = (model.proto.weight @ model.sigma.diag() @ model.proto.weight.T).diagonal().sum()
                elif args['var_type'] == 'anisotropic':
                    wca = (model.proto.weight @ model.sigma @ model.proto.weight.T).diagonal().sum()
                me = torch.log(model.base.dist.entropy().mean())
                loss = loss_func(logits, target) - wca - me
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                # Enforce unit norm on w via projected subgradient method.
                for c in range(args['num_classes']):
                    model.proto.weight.data[c] /= model.proto.weight.data[c].norm()
                # Enforce spectral norm on Sigma and update lower triangular L.
                sigma_svd = torch.svd(model.sigma)
                new_sigma = sigma_svd[0] @ sigma_svd[1].clamp(epsilon, 1).diag() @ sigma_svd[2].T
                new_L = torch.cholesky(new_sigma)
                model.base.L.copy_(new_L)
        train_accuracy = accuracy(model, train_loader, device=device, norm=norm_func)
        test_accuracy = accuracy(model, test_loader, device=device, norm=norm_func)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        sigma_hist.append(model.sigma.detach().cpu().numpy())
        print('Epoch {:03}, Train acc: {:.3f}, Test acc: {:.3f}'.format(epoch + 1, train_acc[-1], test_acc[-1]))
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            model.save(os.path.join(args['output_path']['models'], 'ckpt_best'))
            print('Best accuracy achieved on epoch {}.'.format(epoch + 1))
    np.save(os.path.join(args['output_path']['stats'], 'train_acc.npy'), np.array(train_acc))
    np.save(os.path.join(args['output_path']['stats'], 'test_acc.npy'), np.array(test_acc))
    np.save(os.path.join(args['output_path']['stats'], 'sigma_hist.npy'), np.array(sigma_hist))


def train_stochastic_adversarial(model, train_loader, test_loader, args, device='cpu'):
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
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
    epsilon = 1e-3
    best_test_acc = -1.
    train_acc, test_acc = [], []
    sigma_hist = []
    for epoch in range(args['num_epochs']):
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
            clean_loss = loss_func(logits, target)
            adv_loss = loss_func(adv_logits, target)
            if args['reg_type'] == 'wca':
                if args['var_type'] == 'isotropic':
                    wca = (model.proto.weight @ model.sigma.diag() @ model.proto.weight.T).diagonal().sum()
                elif args['var_type'] == 'anisotropic':
                    wca = (model.proto.weight @ model.sigma @ model.proto.weight.T).diagonal().sum()
                loss = args['w_ct'] * clean_loss + args['w_at'] * adv_loss - wca
            elif args['reg_type'] == 'max_entropy':
                me = torch.log(model.base.dist.entropy().mean())
                loss = args['w_ct'] * clean_loss + args['w_at'] * adv_loss - me
            elif args['reg_type'] == 'wca+max_entropy':
                if args['var_type'] == 'isotropic':
                    wca = (model.proto.weight @ model.sigma.diag() @ model.proto.weight.T).diagonal().sum()
                elif args['var_type'] == 'anisotropic':
                    wca = (model.proto.weight @ model.sigma @ model.proto.weight.T).diagonal().sum()
                me = torch.log(model.base.dist.entropy().mean())
                loss = args['w_ct'] * clean_loss + args['w_at'] * adv_loss - wca - me
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                # Enforce unit norm on w via projected subgradient method.
                for c in range(args['num_classes']):
                    model.proto.weight.data[c] /= model.proto.weight.data[c].norm()
                # Enforce spectral norm on Sigma and update lower triangular L.
                sigma_svd = torch.svd(model.sigma)
                new_sigma = sigma_svd[0] @ sigma_svd[1].clamp(epsilon, 1).diag() @ sigma_svd[2].T
                new_L = torch.cholesky(new_sigma)
                model.base.L.copy_(new_L)
        train_accuracy = accuracy(model, train_loader, device=device, norm=norm_func)
        test_accuracy = accuracy(model, test_loader, device=device, norm=norm_func)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        sigma_hist.append(model.sigma.detach().cpu().numpy())
        print('Epoch {:03}, Train acc: {:.3f}, Test acc: {:.3f}'.format(epoch + 1, train_acc[-1], test_acc[-1]))
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            model.save(os.path.join(args['output_path']['models'], 'ckpt_best'))
            print('Best accuracy achieved on epoch {}.'.format(epoch + 1))
    np.save(os.path.join(args['output_path']['stats'], 'train_acc.npy'), np.array(train_acc))
    np.save(os.path.join(args['output_path']['stats'], 'test_acc.npy'), np.array(test_acc))
    np.save(os.path.join(args['output_path']['stats'], 'sigma_hist.npy'), np.array(sigma_hist))
