import torch
from torchvision import datasets, transforms


def get_data_loader(dataset, batch_size, train):
    if dataset not in ('mnist', 'cifar10'):
        raise NotImplementedError('Dataset not supported.')
    if dataset == 'mnist':
        tr = transforms.Compose([
            transforms.ToTensor(),
        ])
        d = datasets.MNIST('./data', train=train, transform=tr)
    elif dataset == 'cifar10':
        if train:
            tr = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            tr = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        d = datasets.CIFAR10('./data', train=train, transform=tr)
    data_loader = torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader
