import torch
from torchvision import datasets, transforms


def get_data_loader(dataset, batch_size, train=True, shuffle=True, drop_last=True):
    if dataset not in ('mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn'):
        raise NotImplementedError('Dataset not supported.')
    if dataset == 'mnist':
        tr = transforms.Compose([
            transforms.ToTensor(),
        ])
        d = datasets.MNIST('./data', train=train, transform=tr)
    if dataset == 'fmnist':
        tr = transforms.Compose([
            transforms.ToTensor(),
        ])
        d = datasets.FashionMNIST('./data', train=train, transform=tr)
    elif dataset == 'cifar10':
        if train:
            tr = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Line commented out in case we use adv. examples during training.
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            tr = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        d = datasets.CIFAR10('./data', train=train, transform=tr)
    elif dataset == 'cifar100':
        if train:
            tr = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Line commented out in case we use adv. examples during training.
                # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        else:
            tr = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        d = datasets.CIFAR100('./data', train=train, transform=tr)
    elif dataset == 'svhn':
        if train:
            tr = transforms.Compose([
                transforms.ToTensor(),
                # Line commented out in case we use adv. examples during training.
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            tr = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        d = datasets.SVHN('./data', train=train, transform=tr)
    data_loader = torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return data_loader
