import json
import os
import sys

import torch
from foolbox import PyTorchModel
from foolbox.attacks import FGSMMC, PGDMC, BIMMC

from data_loaders import get_data_loader
from models import model_factory
from train import train_vanilla, train_stochastic


def parse_args():
    mode = sys.argv[1]
    if mode not in ('train', 'test', 'train+test'):
        raise ValueError()
    config_file = sys.argv[2]
    with open(config_file, 'r') as fp:
        args = json.loads(fp.read().strip())
    return mode, args


def train(args, device):
    os.makedirs(args['output_path']['stats'], exist_ok=True)
    os.makedirs(args['output_path']['models'], exist_ok=True)
    train_loader = get_data_loader(args['dataset'], args['batch_size'], train=True, shuffle=True, drop_last=True)
    test_loader = get_data_loader(args['dataset'], args['batch_size'], train=False, shuffle=False, drop_last=False)
    model = model_factory(args['dataset'], args['training_type'], args['feature_dim'])
    model.to(device)
    if args['training_type'] == 'vanilla':
        print('Vanilla training.')
        train_vanilla(model, train_loader, test_loader, args, device=device)
    elif args['training_type'] == 'stochastic':
        print('Stochastic training.')
        train_stochastic(model, train_loader, test_loader, args, device=device)
    else:
        raise NotImplementedError('Training "{}" not implemented. Supported: [vanilla|stochastic].'.format(
            args['training_type']))
    print('Finished training.')


def test(args, device):
    model = model_factory(args['dataset'], args['training_type'], args['feature_dim'])
    model.to(device)
    model.load(os.path.join(args['output_path']['models'], 'ckpt_best'))
    model.eval()
    if args['dataset'] == 'cifar10':
        preprocessing = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), axis=-3)
    else:
        preprocessing = None
    fbox_model = PyTorchModel(model, bounds=(0, 1), device=device, preprocessing=preprocessing)
    test_loader = get_data_loader(args['dataset'], args['batch_size'], False, shuffle=False, drop_last=False)
    attacks = [
        FGSMMC(),
        PGDMC(rel_stepsize=0.1, steps=10),
        BIMMC(rel_stepsize=0.1),
    ]
    attack_names = ['FGSM', 'PGD', 'BIM']
    eps_names = ['  0/255', '  1/255', '  2/255', '  4/255', '  8/255', ' 16/255', ' 32/255', ' 64/255', '128/255']
    eps_values = [0. / 255, 1. / 255, 2. / 255, 4. / 255, 8. / 255, 16. / 255, 32. / 255, 64. / 255, 128. / 255]
    print('Adversarial testing.')
    for idx, attack in enumerate(attacks):
        success_cum = []
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            advs, _, success = attack(fbox_model, data, target, epsilons=eps_values, mc=args['monte_carlo_runs'])
            success_cum.append(success)
        success_cum = torch.cat(success_cum, dim=1)
        robust_accuracy = 1 - success_cum.float().mean(axis=-1)
        print('Attack: {}'.format(attack_names[idx]))
        for eps_name, eps_value, accuracy in zip(eps_names, eps_values, robust_accuracy):
            print('ε: {}, acc: {:.3f}'.format(eps_name, accuracy.item()))
    print('Finished testing.')


def main(mode, args):
    if args['device'] is not None:
        device = args['device']
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args)
    if mode == 'train':
        train(args, device)
    elif mode == 'test':
        test(args, device)
    else:
        train(args, device)
        test(args, device)


if __name__ == '__main__':
    try:
        mode, args = parse_args()
    except ValueError:
        print('Invalid mode. Usage: python run.py <mode[train|test|train+test]> <config. file>')
    except IndexError:
        print('Path to configuration file missing. Usage: python run.py <mode[train|test|train+test]> <config. file>')
        sys.exit()
    except FileNotFoundError:
        print('Incorrect path to configuration file. File not found.')
        sys.exit()
    except json.JSONDecodeError:
        print('Configuration file is an invalid JSON.')
        sys.exit()
    main(mode, args)