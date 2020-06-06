import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from metrics import accuracy


def train_adv(model, train_loader, test_loader, attack, args, device='cpu'):
    optimizer = Adam(model.parameters(), lr=args['lr'])
    loss_func = nn.CrossEntropyLoss()
    noise_entropy_threshold = math.log(args['var_threshold']) + (1 + math.log(2 * math.pi)) / 2
    best_test_acc = -1.
    train_accuracy = []
    test_accuracy = []
    for epoch in range(args['num_epochs']):
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            # Apply the attack to generate perturbed data.
            perturbed_data = attack(model, data, target).to(device)
            model.train()
            # Compute logits for clean and perturbed data.
            logits_clean = model(data)
            logits_adv = model(perturbed_data)
            optimizer.zero_grad()
            # Compute the cross-entropy loss for these logits.
            clean_loss = loss_func(logits_clean, target)
            adv_loss = loss_func(logits_adv, target)
            noise_entropy = torch.relu(noise_entropy_threshold - model.noise.dist.entropy()).mean()
            # Balance these two losses with weight w, and add the regularization term.
            w = args['adv_loss_w']
            loss = w * adv_loss + (1. - w) * clean_loss + args['reg_term'] * noise_entropy
            loss.backward()
            optimizer.step()
        train_accuracy.append(accuracy(model, train_loader, device=device))
        test_accuracy.append(accuracy(model, test_loader, device=device))
        # Checkpoint current model.
        torch.save(model.state_dict(), os.path.join(args['output_path']['models'], 'ckpt.pt'))
        # Save model with best testing performance.
        if test_accuracy[-1] > best_test_acc:
            best_test_acc = test_accuracy[-1]
            torch.save(model.state_dict(), os.path.join(args['output_path']['models'], 'ckpt_best.pt'))
        print('Epoch {}\t\tTrain acc: {:.3f}, Test acc: {:.3f}'.format(
            epoch + 1, train_accuracy[-1], test_accuracy[-1]))
    # Also save the training and testing curves.
    np.save(os.path.join(args['output_path']['stats'], 'train_acc.npy'), np.array(train_accuracy))
    np.save(os.path.join(args['output_path']['stats'], 'test_acc.npy'), np.array(test_accuracy))


def meta_train_adv(model, train_loader, test_loader, attack, args, device='cpu'):
    optim_inner = Adam([
        {'params': model.gen.parameters()},
        {'params': model.noise.fc_mu.parameters()},
        {'params': model.noise.fc_sigma.parameters()},
        {'params': model.proto.parameters()},
    ], lr=args['lr'])
    optim_outer = Adam([
        {'params': model.noise.fc_b.parameters()},
        {'params': model.reg_term.parameters()},
    ], lr=args['meta_lr'])
    loss_func = nn.CrossEntropyLoss()
    noise_entropy_threshold = math.log(args['var_threshold']) + (1 + math.log(2 * math.pi)) / 2
    best_test_acc = -1.
    train_accuracy = []
    test_accuracy = []
    for epoch in range(args['num_epochs']):
        for data_outer, target_outer in train_loader:
            for data_inner, target_inner in train_loader:
                if epoch < args['meta_epoch']:
                    break
                data_inner = data_inner.to(device)
                target_inner = target_inner.to(device)
                # Apply the attack to generate perturbed data.
                perturbed_data = attack(model, data_inner, target_inner).to(device)
                model.train()
                # Compute logits for clean and perturbed data.
                logits_clean = model(data_inner)
                logits_adv = model(perturbed_data)
                optim_inner.zero_grad()
                # Compute the cross-entropy loss for these logits.
                clean_loss = loss_func(logits_clean, target_inner)
                adv_loss = loss_func(logits_adv, target_inner)
                noise_entropy = torch.relu(noise_entropy_threshold - model.noise.dist.entropy()).mean()
                # Balance these two losses with weight w, and add the regularization term.
                w = args['adv_loss_w']
                loss = w * adv_loss + (1. - w) * clean_loss + model.get_reg_term() * noise_entropy
                loss.backward()
                optim_inner.step()
            data_outer = data_outer.to(device)
            target_outer = target_outer.to(device)
            # Apply the attack to generate perturbed data.
            perturbed_data = attack(model, data_outer, target_outer).to(device)
            model.train()
            # Compute logits for clean and perturbed data.
            logits_clean = model(data_outer)
            logits_adv = model(perturbed_data)
            optim_outer.zero_grad()
            # Compute the cross-entropy loss for these logits.
            clean_loss = loss_func(logits_clean, target_outer)
            adv_loss = loss_func(logits_adv, target_outer)
            noise_entropy = torch.relu(noise_entropy_threshold - model.noise.dist.entropy()).mean()
            # Balance these two losses with weight w, and add the regularization term.
            w = args['adv_loss_w']
            loss = w * adv_loss + (1. - w) * clean_loss + model.get_reg_term() * noise_entropy
            loss.backward()
            optim_outer.step()
        train_accuracy.append(accuracy(model, train_loader, device=device))
        test_accuracy.append(accuracy(model, test_loader, device=device))
        # Checkpoint current model.
        torch.save(model.state_dict(), os.path.join(args['output_path']['models'], 'ckpt.pt'))
        # Save model with best testing performance.
        if test_accuracy[-1] > best_test_acc:
            best_test_acc = test_accuracy[-1]
            torch.save(model.state_dict(), os.path.join(args['output_path']['models'], 'ckpt_best.pt'))
        print('Epoch {}\t\tTrain acc: {:.3f}, Test acc: {:.3f}'.format(
            epoch + 1, train_accuracy[-1], test_accuracy[-1]))
    # Also save the training and testing curves.
    np.save(os.path.join(args['output_path']['stats'], 'train_acc.npy'), np.array(train_accuracy))
    np.save(os.path.join(args['output_path']['stats'], 'test_acc.npy'), np.array(test_accuracy))
