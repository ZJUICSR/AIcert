import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms

TORCH_HUB_DIR = '/vulcanscratch/psando/TorchHub'

IMAGENET_VAL_DIR = '/vulcanscratch/psando'
VOC_VAL_DIR = '/vulcanscratch/psando/VOC'
CIFAR_VAL_DIR = '/fs/vulcan-datasets/CIFAR'

debug = False

def get_model(model_name, device):
    torch.hub.set_dir(TORCH_HUB_DIR)
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)

    model = model.to(device)
    return model

def normalize(x):
    """
    Normalizes a batch of images with size (batch_size, 3, height, width)
    by mean and std dev expected by PyTorch models
    """
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    return (x - mean.type_as(x)[None,:,None,None]) / std.type_as(x)[None,:,None,None]

def get_conv_layers(model):
    return [module for module in model.modules() if type(module) == nn.Conv2d]

def l2_layer_loss(model, delta):
    loss = torch.tensor(0.)
    activations = []
    remove_handles = []

    def activation_recorder_hook(self, input, output):
        activations.append(output)
        return None

    for conv_layer in get_conv_layers(model):
        handle = conv_layer.register_forward_hook(activation_recorder_hook)
        remove_handles.append(handle)

    model.eval()
    model.zero_grad()
    model(delta)

    # unregister hook so activation tensors have no references
    for handle in remove_handles:
        handle.remove()

    loss = -sum(list(map(lambda activation: torch.log(torch.sum(torch.square(activation)) / 2), activations)))
    return loss

def get_index_to_label_map(dataset_name):
    if dataset_name == 'imagenet':
        with open('imagenet_class_index.json', 'r') as read_file:
            class_idx = json.load(read_file)
            index_to_label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            return index_to_label

def get_data_loader(dataset_name, batch_size=64, shuffle=False):
    """
    Returns a DataLoader with validation images for dataset_name
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    if dataset_name == 'imagenet':
        val_dataset = datasets.ImageNet(IMAGENET_VAL_DIR, split='val', transform=transform)
    elif dataset_name == 'cifar10':
        val_dataset = datasets.CIFAR10(CIFAR_VAL_DIR, train=False, transform=transform)
    elif dataset_name == 'voc2012':
        val_dataset = datasets.VOCDetection(VOC_VAL_DIR, year='2012', image_set='val', transform=transform)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return val_loader


def get_fooling_rate(model, delta, data_loader, device, disable_tqdm=False):
    """
    Computes the fooling rate of the UAP delta on the dataset.
    Fooling rate is a measure of change in the model's output
    caused by the perturbation. In this case, fooling rate is
    the proportion of outputs that are changed by adding delta.
    Ex. delta = torch.zeros() should have a fooling rate of 0.0
    """
    flipped = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, disable=disable_tqdm):
            images, labels = batch
            images = images.to(device)
            adv_images = torch.add(delta, images).clamp(0, 1)

            outputs = model(normalize(images))
            adv_outputs = model(normalize(adv_images))
            _, predicted = torch.max(outputs.data, 1)
            _, adv_predicted = torch.max(adv_outputs.data, 1)

            total += images.size(0)
            flipped += (predicted != adv_predicted).sum().item()

    return flipped / total

def get_baseline_fooling_rate(model, device, disable_tqdm=False):
    """
    Baseline fooling rate is always evaluated on ILSVRC 2012 dataset
    """
    xi_min = -10/255
    xi_max = 10/255
    delta = (xi_min - xi_max) * torch.rand((1, 3, 224, 224), device=device) + xi_max
    delta.requires_grad = True
    data_loader = get_data_loader('imagenet')
    fr = get_fooling_rate(model, delta, data_loader, device, disable_tqdm=disable_tqdm)
    return fr

def get_rate_of_saturation(delta, xi):
    """
    Returns the proportion of pixels in delta
    that have reached the max-norm limit xi
    """
    return np.sum(np.equal(np.abs(delta), xi)) / np.size(delta)

def gd_universal_adversarial_perturbation(model, model_name, train_type, batch_size, device, dataset_name, patience_interval, id, disable_tqdm=False):
    """
    Returns a universal adversarial perturbation tensor
    """

    max_iter = 10000
    size = 224

    sat_threshold = 0.00001
    sat_prev = 0
    sat = 0
    sat_change = 0
    sat_min = 0.5
    sat_should_rescale = False

    iter_since_last_fooling = 0
    iter_since_last_best = 0
    best_fooling_rate = 0

    xi_min = -10/255
    xi_max = 10/255
    delta = (xi_min - xi_max) * torch.rand((1, 3, size, size), device=device) + xi_max
    delta.requires_grad = True

    print(f"Initial norm: {torch.norm(delta, p=np.inf)}")

    optimizer = optim.Adam([delta], lr=0.1)
    data_loader = get_data_loader(dataset_name, batch_size=batch_size)

    for i in tqdm(range(max_iter), disable=disable_tqdm):
        iter_since_last_fooling += 1
        optimizer.zero_grad()
        loss = l2_layer_loss(model, delta)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"Iter {i}, Loss: {loss.item()}")
            if debug:
                print(f"Norm before clip: {torch.norm(delta, p=np.inf)}")

        # clip delta after each step
        with torch.no_grad():
            delta.clamp_(xi_min, xi_max)

        # compute rate of saturation on a clamped delta
        sat_prev = np.copy(sat)
        sat = get_rate_of_saturation(delta.cpu().detach().numpy(), xi_max)
        sat_change = np.abs(sat - sat_prev)

        if sat_change < sat_threshold and sat > sat_min:
            if debug:
                print(f"Saturated delta in iter {i} with {sat} > {sat_min}\nChange in saturation: {sat_change} < {sat_threshold}\n")
            sat_should_rescale = True

        # fooling rate is measured every 200 iterations if saturation threshold is crossed
        # otherwise, fooling rate is measured every 400 iterations
        if iter_since_last_fooling > 400 or (sat_should_rescale and iter_since_last_fooling > 200):
            iter_since_last_fooling = 0
            print("Getting latest fooling rate...")
            current_fooling_rate = get_fooling_rate(model, delta, data_loader, device, disable_tqdm=disable_tqdm)
            print(f"Latest fooling rate: {current_fooling_rate}")

            if current_fooling_rate > best_fooling_rate:
                print(f"Best fooling rate thus far: {current_fooling_rate}")
                best_fooling_rate = current_fooling_rate
                filename = f"perturbations/{id}_{model_name}_{train_type}_iter={i}_val={dataset_name}_fr={int(best_fooling_rate * 1000)}"
                # np.save(filename, delta.cpu().detach().numpy())
            else:
                iter_since_last_best += 1

            # if the best fooling rate has not been overcome after patience_interval iterations
            # then training is considered complete
            if iter_since_last_best == patience_interval:
                break

        if sat_should_rescale:
            with torch.no_grad():
                delta.data = delta.data / 2
            sat_should_rescale = False

    print(f"Training complete.\nLast delta saved at: {filename}\nLast delta Iter: {i}, Loss: {loss}, Fooling rate: {best_fooling_rate}")
    return delta
