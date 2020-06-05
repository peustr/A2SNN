import json
import os
import sys

import torch

from attacks import fgsm, pgd
from data_loaders import get_data_loader
from models import model_factory
from train import train_adv, meta_train_adv


def parse_args():
    config_file = sys.argv[1]
    with open(config_file, 'r') as fp:
        args = json.loads(fp.read().strip())
    return args


def main(args):
    os.makedirs(args['output_paths']['stats'], exist_ok=True)
    os.makedirs(args['output_paths']['models'], exist_ok=True)
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
