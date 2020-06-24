import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from metrics import accuracy
from utils import normalize_cifar10


def train_adv(model, train_loader, test_loader, attack, args, device='cpu'):
    optimizer = Adam([
        {'params': model.gen.parameters()},
        {'params': model.sigma},
        {'params': model.proto.parameters()},
    ], lr=args['lr'])
    loss_func = nn.CrossEntropyLoss()
    if args['dataset'] == 'cifar10':
        norm_func = normalize_cifar10
    else:
        norm_func = None
    best_test_acc = -1.
    train_accuracy = []
    test_accuracy = []
    sigma_hist = []
    for epoch in range(args['num_epochs']):
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            # Apply the attack to generate perturbed data.
            if isinstance(args['epsilon'], float):
                perturbed_data = attack(model, data, target, epsilon=args['epsilon']).to(device)
            elif args['epsilon'] == 'rand':
                rand_epsilon = np.random.choice([8. / 255., 16. / 255., 32. / 255., 64. / 255., 128. / 255.])
                perturbed_data = attack(model, data, target, epsilon=rand_epsilon).to(device)
            else:
                perturbed_data = attack(model, data, target).to(device)
            model.train()
            # Compute logits for clean and perturbed data.
            if norm_func is not None:
                data = norm_func(data)
                perturbed_data = norm_func(perturbed_data)
            logits_clean = model(data)
            logits_adv = model(perturbed_data)
            optimizer.zero_grad()
            # Compute the cross-entropy loss for these logits.
            clean_loss = loss_func(logits_clean, target)
            adv_loss = loss_func(logits_adv, target)
            threshold = math.log(args['var_threshold']) + (1 + math.log(2 * math.pi)) / 2
            noise_entropy = torch.relu(threshold - model.dist.entropy()).mean()
            # Balance these two losses with weight w, and add the regularization term.
            w = args['adv_loss_w']
            loss = w * adv_loss + (1. - w) * clean_loss + args['reg_term'] * noise_entropy
            loss.backward()
            optimizer.step()
        train_accuracy.append(accuracy(model, train_loader, device=device, norm=norm_func))
        test_accuracy.append(accuracy(model, test_loader, device=device, norm=norm_func))
        sigma_hist.append(model.sigma.detach().cpu().numpy())
        # Checkpoint current model.
        model.save(os.path.join(args['output_path']['models'], 'ckpt'))
        # Save model with best testing performance.
        if test_accuracy[-1] > best_test_acc:
            best_test_acc = test_accuracy[-1]
            model.save(os.path.join(args['output_path']['models'], 'ckpt_best'))
        print('Epoch {}\t\tTrain acc: {:.3f}, Test acc: {:.3f}'.format(
            epoch + 1, train_accuracy[-1], test_accuracy[-1]))
    # Also save the training and testing curves.
    np.save(os.path.join(args['output_path']['stats'], 'train_acc.npy'), np.array(train_accuracy))
    np.save(os.path.join(args['output_path']['stats'], 'test_acc.npy'), np.array(test_accuracy))
    np.save(os.path.join(args['output_path']['stats'], 'sigma_hist.npy'), np.array(sigma_hist))


def meta_train_adv(model, train_loader, val_loader, test_loader, attack, args, device='cpu'):
    optim_inner = Adam([
        {'params': model.gen.parameters()},
        {'params': model.sigma},
        {'params': model.proto.parameters()},
    ], lr=args['lr'])
    optim_outer = Adam([
        {'params': model.b},
        {'params': model.lambda2},
    ], lr=args['meta_lr'])
    loss_func_inner = nn.CrossEntropyLoss(reduction='mean')
    loss_func_outer = nn.CrossEntropyLoss(reduction='mean')
    if args['dataset'] == 'cifar10':
        norm_func = normalize_cifar10
    else:
        norm_func = None
    best_test_acc = -1.
    train_accuracy = []
    test_accuracy = []
    sigma_hist, b_hist, lambda2_hist = [], [], []
    val_iter = iter(val_loader)
    for epoch in range(args['num_epochs']):
        for data_inner, target_inner in train_loader:
            data_inner = data_inner.to(device)
            target_inner = target_inner.to(device)
            # Apply the attack to generate perturbed data.
            if isinstance(args['epsilon'], float):
                perturbed_data_inner = attack(model, data_inner, target_inner, epsilon=args['epsilon']).to(device)
            elif args['epsilon'] == 'rand':
                rand_epsilon = np.random.choice([8. / 255., 16. / 255., 32. / 255., 64. / 255., 128. / 255.])
                perturbed_data_inner = attack(model, data_inner, target_inner, epsilon=rand_epsilon).to(device)
            else:
                perturbed_data_inner = attack(model, data_inner, target_inner).to(device)
            model.train()
            # Compute logits for clean and perturbed data.
            if norm_func is not None:
                data_inner = norm_func(data_inner)
                perturbed_data_inner = norm_func(perturbed_data_inner)
            logits_clean_inner = model(data_inner)
            logits_adv_inner = model(perturbed_data_inner)
            optim_inner.zero_grad()
            # Compute the cross-entropy loss for these logits.
            clean_loss_inner = loss_func_inner(logits_clean_inner, target_inner)
            adv_loss_inner = loss_func_inner(logits_adv_inner, target_inner)
            threshold_inner = model.get_b() + (1. + math.log(2. * math.pi)) / 2.
            noise_entropy_inner = torch.relu(threshold_inner - model.dist.entropy()).mean()
            # Balance these two losses with weight w, and add the regularization term.
            w = args['adv_loss_w']
            loss_inner = w * adv_loss_inner + (1. - w) * clean_loss_inner + model.get_lambda2() * noise_entropy_inner
            loss_inner.backward()
            optim_inner.step()
            try:
                data_outer, target_outer = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                data_outer, target_outer = next(val_iter)
            data_outer = data_outer.to(device)
            target_outer = target_outer.to(device)
            # Apply the attack to generate perturbed data.
            if isinstance(args['epsilon'], float):
                perturbed_data_outer = attack(model, data_outer, target_outer, epsilon=args['epsilon']).to(device)
            elif args['epsilon'] == 'rand':
                rand_epsilon = np.random.choice([8. / 255., 16. / 255., 32. / 255., 64. / 255., 128. / 255.])
                perturbed_data_outer = attack(model, data_outer, target_outer, epsilon=rand_epsilon).to(device)
            else:
                perturbed_data_outer = attack(model, data_outer, target_outer).to(device)
            model.train()
            # Compute logits for clean and perturbed data.
            if norm_func is not None:
                data_outer = norm_func(data_outer)
                perturbed_data_outer = norm_func(perturbed_data_outer)
            logits_clean_outer = model(data_outer)
            logits_adv_outer = model(perturbed_data_outer)
            optim_outer.zero_grad()
            # Compute the cross-entropy loss for these logits.
            clean_loss_outer = loss_func_outer(logits_clean_outer, target_outer)
            adv_loss_outer = loss_func_outer(logits_adv_outer, target_outer)
            threshold_outer = model.get_b() + (1. + math.log(2. * math.pi)) / 2.
            noise_entropy_outer = torch.relu(threshold_outer - model.dist.entropy()).mean()
            # Balance these two losses with weight w, and add the regularization term.
            w = args['adv_loss_w']
            loss_outer = w * adv_loss_outer + (1. - w) * clean_loss_outer + model.get_lambda2() * noise_entropy_outer
            loss_outer.backward()
            optim_outer.step()
        train_accuracy.append(accuracy(model, train_loader, device=device, norm=norm_func))
        test_accuracy.append(accuracy(model, test_loader, device=device, norm=norm_func))
        sigma_hist.append(model.sigma.detach().cpu().numpy())
        b_hist.append(model.get_b().detach().cpu().numpy())
        lambda2_hist.append(model.get_lambda2().detach().cpu().numpy())
        # Checkpoint current model.
        model.save(os.path.join(args['output_path']['models'], 'ckpt'))
        # Save model with best testing performance.
        if test_accuracy[-1] > best_test_acc:
            best_test_acc = test_accuracy[-1]
            model.save(os.path.join(args['output_path']['models'], 'ckpt_best'))
        print('Epoch {}\t\tTrain acc: {:.3f}, Test acc: {:.3f}'.format(
            epoch + 1, train_accuracy[-1], test_accuracy[-1]))
    # Also save the training and testing curves.
    np.save(os.path.join(args['output_path']['stats'], 'train_acc.npy'), np.array(train_accuracy))
    np.save(os.path.join(args['output_path']['stats'], 'test_acc.npy'), np.array(test_accuracy))
    np.save(os.path.join(args['output_path']['stats'], 'sigma_hist.npy'), np.array(sigma_hist))
    np.save(os.path.join(args['output_path']['stats'], 'b_hist.npy'), np.array(b_hist))
    np.save(os.path.join(args['output_path']['stats'], 'lambda2_hist.npy'), np.array(lambda2_hist))
