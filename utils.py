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
