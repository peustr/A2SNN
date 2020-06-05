import torch
import torch.nn.functional as f


def fgsm(model, data, target, epsilon=8./255., data_min=0, data_max=1):
    model.eval()

    perturbed_data = data.clone()
    perturbed_data.requires_grad = True

    output = model(perturbed_data)

    loss = f.cross_entropy(output, target)
    if perturbed_data.grad is not None:
        perturbed_data.grad.data.zero_()
    loss.backward()

    sign_data_grad = perturbed_data.grad.data.sign()
    with torch.no_grad():
        perturbed_data += epsilon * sign_data_grad
        perturbed_data.clamp_(data_min, data_max)

    perturbed_data.requires_grad = False
    return perturbed_data


def pgd(model, data, target, k=7, epsilon=8./255., a=0.01, d_min=0, d_max=1):
    model.eval()

    perturbed_data = data.clone()
    perturbed_data.requires_grad = True

    data_max = data + epsilon
    data_min = data - epsilon
    data_max.clamp_(d_min, d_max)
    data_min.clamp_(d_min, d_max)

    with torch.no_grad():
        perturbed_data.data = data + perturbed_data.uniform_(-1. * epsilon, epsilon)
        perturbed_data.data.clamp_(d_min, d_max)

    for _ in range(k):
        output = model(perturbed_data)

        loss = f.cross_entropy(output, target)
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()
        loss.backward()

        data_grad = perturbed_data.grad.data
        with torch.no_grad():
            perturbed_data.data += a * torch.sign(data_grad)
            perturbed_data.data = torch.max(torch.min(perturbed_data, data_max), data_min)

    perturbed_data.requires_grad = False
    return perturbed_data
