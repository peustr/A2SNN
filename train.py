import json
import math
import sys

import torch
import torch.nn as nn
from torch.optim import Adam

from attacks import fgsm, pgd
from data_loaders import get_data_loader
from metrics import accuracy
from models import model_factory


def parse_args():
    config_file = sys.argv[1]
    with open(config_file, 'r') as fp:
        args = json.loads(fp.read().strip())
    return args


def train_adv(model, train_loader, test_loader, attack, args, device='cpu'):
    optimizer = Adam(model.parameters(), lr=args['lr'])
    loss_func = nn.CrossEntropyLoss()
    noise_entropy_threshold = math.log(args['var_threshold']) + (1 + math.log(2 * math.pi)) / 2
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
    train_accuracy = accuracy(model, train_loader, device=device)
    test_accuracy = accuracy(model, test_loader, device=device)
    print('Epoch {}\t\tTrain acc: {:.3f}, Test acc: {:.3f}'.format(epoch + 1, train_accuracy, test_accuracy))


def meta_train_adv(model, train_loader, test_loader, attack, args, device='cpu'):
    pass


def main(args):
    if args['device'] is not None:
        device = args['device']
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = get_data_loader(args['dataset'], args['batch_size'], True)
    test_loader = get_data_loader(args['dataset'], args['batch_size'], False)
    model = model_factory(args['dataset'], args['meta_train'], device=device)
    model.to(device)
    if args['attack'] == 'fgsm':
        attack = fgsm
    elif args['attack'] == 'pgd':
        attack = pgd
    else:
        raise NotImplementedError('Attack {} not implemented.'.format(args['attack']))
    if not args['meta_train']:
        train_adv(model, train_loader, test_loader, attack, args, device=device)
    else:
        meta_train_adv(model, train_loader, test_loader, attack, args, device=device)
    print('Finished training.')


if __name__ == '__main__':
    try:
        args = parse_args()
    except IndexError:
        print('Path to config file missing. Usage: python train.py <path to config>')
        sys.exit()
    except FileNotFoundError:
        print('Incorrect path to config file. File not found.')
        sys.exit()
    except json.JSONDecodeError:
        print('Config file is an invalid JSON.')
        sys.exit()
    main(args)
