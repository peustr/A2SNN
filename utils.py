mean_cifar10 = (0.4914, 0.4822, 0.4465)
std_cifar10 = (0.2023, 0.1994, 0.2010)

mean_cifar100 = (0.5071, 0.4867, 0.4408)
std_cifar100 = (0.2675, 0.2565, 0.2761)

mean_generic = (0.5, 0.5, 0.5)
std_generic = (0.5, 0.5, 0.5)


def normalize_cifar10(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean_cifar10[0]) / std_cifar10[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean_cifar10[1]) / std_cifar10[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean_cifar10[2]) / std_cifar10[2]
    return t


def normalize_cifar100(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean_cifar100[0]) / std_cifar100[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean_cifar100[1]) / std_cifar100[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean_cifar100[2]) / std_cifar100[2]
    return t


def normalize_generic(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean_generic[0]) / std_generic[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean_generic[1]) / std_generic[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean_generic[2]) / std_generic[2]
    return t


dataset_param_mapping = {
    'mnist': {
        'e_des': ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5'],
        'e_val': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    },
    'fmnist': {
        'e_des': ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5'],
        'e_val': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    },
    'cifar10': {
        'e_des': ['  0/255', '  1/255', '  2/255', '  4/255', '  8/255', ' 16/255', ' 32/255', ' 64/255', '128/255'],
        'e_val': [0. / 255, 1. / 255, 2. / 255, 4. / 255, 8. / 255, 16. / 255, 32. / 255, 64. / 255, 128. / 255],
    },
    'cifar100': {
        'e_des': ['  0/255', '  1/255', '  2/255', '  4/255', '  8/255', ' 16/255', ' 32/255', ' 64/255', '128/255'],
        'e_val': [0. / 255, 1. / 255, 2. / 255, 4. / 255, 8. / 255, 16. / 255, 32. / 255, 64. / 255, 128. / 255],
    },
    'svhn': {
        'e_des': ['  0/255', '  1/255', '  2/255', '  4/255', '  8/255', ' 16/255', ' 32/255', ' 64/255', '128/255'],
        'e_val': [0. / 255, 1. / 255, 2. / 255, 4. / 255, 8. / 255, 16. / 255, 32. / 255, 64. / 255, 128. / 255],
    },
}


attack_param_mapping = {
    'FGSM': dataset_param_mapping,
    'PGD': dataset_param_mapping,
    'BIM': dataset_param_mapping,
    'C&W': {
        'cifar10': {
            'e_des': ['None'],
            'e_val': [None],
        },
    },
    'Few-Pixel': {
        'cifar10': {
            'e_des': ['1 pixel', '2 pixels', '3 pixels'],
            'e_val': [1, 2, 3],
        },
    },
}
