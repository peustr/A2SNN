import torch

from foolbox import PyTorchModel
from foolbox.attacks import FGSMMC, PGDMC, BIMMC, CW, BoundaryAttack


attacks = {
    'FGSM': FGSMMC(),
    'PGD': PGDMC(rel_stepsize=1.0, steps=10),
    'BIM': BIMMC(rel_stepsize=1.0, steps=10),
    'C&W': CW(),
    'Boundary Attack': BoundaryAttack(),
}


def test_attack(model, data_loader, attack_name, epsilon_values, args, device='cpu'):
    model.eval()
    attack_model = attacks[attack_name]
    if args['dataset'] == 'mnist':
        preprocessing = None
    elif args['dataset'] == 'cifar10':
        preprocessing = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), axis=-3)
    else:
        raise NotImplementedError('Dataset not supported.')
    fbox_model = PyTorchModel(model, bounds=(0, 1), device=device, preprocessing=preprocessing)
    success_cum = []
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        if attack_name in ('FGSM', 'PGD', 'BIM'):
            advs, _, success = attack_model(
                fbox_model, data, target, epsilons=epsilon_values, mc=args['monte_carlo_runs'])
        elif attack_name in ('C&W', 'Boundary Attack'):
            advs, _, success = attack_model(fbox_model, data, target, epsilons=epsilon_values)
        success_cum.append(success)
    success_cum = torch.cat(success_cum, dim=1)
    robust_accuracy = 1 - success_cum.float().mean(axis=-1)
    return robust_accuracy
