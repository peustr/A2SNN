import json
import os
import sys

import torch
from foolbox import PyTorchModel
from foolbox.attacks import FGSMMC, BIMMC, PGDMC

from data_loaders import get_data_loader
from models import model_factory


def parse_args():
    config_file = sys.argv[1]
    with open(config_file, 'r') as fp:
        args = json.loads(fp.read().strip())
    return args


def main(args):
    print(args)
    if args['device'] is not None:
        device = args['device']
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_factory(args['dataset'], args['meta_train'], device=device)
    model.to(device)
    ckpt_best = os.path.join(args['output_path']['models'], 'ckpt_best.pt')
    model.load_state_dict(torch.load(ckpt_best))
    model.eval()
    if args['dataset'] == 'cifar10':
        preprocessing = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), axis=-3)
    else:
        preprocessing = None
    fbox_model = PyTorchModel(model, bounds=(0, 1), device=device, preprocessing=preprocessing)
    test_loader = get_data_loader(args['dataset'], args['batch_size'], False, shuffle=False, drop_last=False)
    attacks = [
        FGSMMC(),
        BIMMC(rel_stepsize=0.1),
        PGDMC(rel_stepsize=0.1, steps=10),
    ]
    epsilons = [0. / 255, 1. / 255, 2. / 255, 4. / 255, 8. / 255, 16. / 255, 32. / 255, 64. / 255, 128. / 255]
    for attack in attacks:
        success_cum = []
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            advs, _, success = attack(fbox_model, data, target, epsilons=epsilons, mc=args['mc'])
            success_cum.append(success)
        success_cum = torch.cat(success_cum, dim=1)
        robust_accuracy = 1 - success_cum.float().mean(axis=-1)
        print('Attack: {}'.format(attack))
        for eps, accuracy in zip(epsilons, robust_accuracy):
            print('epsilon: {}\t\t accuracy: {}'.format(eps, accuracy.item()))


if __name__ == '__main__':
    try:
        args = parse_args()
    except IndexError:
        print('Path to config file missing. Usage: python test.py <path to config>')
        sys.exit()
    except FileNotFoundError:
        print('Incorrect path to config file. File not found.')
        sys.exit()
    except json.JSONDecodeError:
        print('Config file is an invalid JSON.')
        sys.exit()
    main(args)
