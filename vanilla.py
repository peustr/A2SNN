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
    train_accuracy = []
    test_accuracy = []
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
        train_accuracy.append(accuracy(model, train_loader, device=device, norm=norm_func))
        test_accuracy.append(accuracy(model, test_loader, device=device, norm=norm_func))
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
