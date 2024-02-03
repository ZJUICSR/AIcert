# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy
from tqdm import tqdm
from os.path import join, dirname
from .utils import *
from .dataloader import *
from .model import create_model
from .optimizer import *


def generate_adv_dataset(model, device='cuda'):
    adv_train_dataset = adv_dataset()

    model = model.eval()
    atk_model = torchattacks.PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
    transform_test = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)

    for images, labels in trainloader:

        images = images.to(device)
        labels = labels.to(device)
        
        adv_images = atk_model(images, labels)  
        adv_train_dataset.append_data(adv_images, labels)

    return adv_train_dataset


def layer_sharpness(model, epsilon=0.1, device='cuda'):
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])

    model = nn.Sequential(norm_layer, model).to(device)
    
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(generate_adv_dataset(deepcopy(model), device=device), batch_size=512, shuffle=True, num_workers=0)
    origin_total = 0
    origin_loss = 0.0
    origin_acc = 0
    with torch.no_grad():
        model.eval()
        for inputs, targets in trainloader:
            outputs = model(inputs)
            origin_total += targets.shape[0]
            origin_loss += criterion(outputs, targets).item() * targets.shape[0]
            _, predicted = outputs.max(1)
            origin_acc += predicted.eq(targets).sum().item()        
        
        origin_acc /= origin_total
        origin_loss /= origin_total

    print("{:35}, Robust Loss: {:10.2f}, Robust Acc: {:10.2f}".format("Origin", origin_loss, origin_acc*100))

    model.eval()
    layer_sharpness_dict = {} 
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # print(name)
            # For WideResNet
            if "sub" in name:
                continue
            layer_sharpness_dict[name] = 1e10

    for layer_name, _ in model.named_parameters():
        if "weight" in layer_name and layer_name[:-len(".weight")] in layer_sharpness_dict.keys():
            # print(layer_name)
            cloned_model = deepcopy(model)
            # set requires_grad sign for each layer
            for name, param in cloned_model.named_parameters():
                # print(name)
                if name == layer_name:
                    # print(name)
                    param.requires_grad = True
                    init_param = param.detach().clone()
                else:
                    param.requires_grad = False
        
            optimizer = torch.optim.SGD(cloned_model.parameters(), lr=1)

            max_loss = 0.0
            min_acc = 0
    
            for epoch in range(10):
                # Gradient ascent
                for inputs, targets in trainloader:
                    optimizer.zero_grad()
                    outputs = cloned_model(inputs)
                    loss = -1 * criterion(outputs, targets) 
                    loss.backward()
                    optimizer.step()
                sd = cloned_model.state_dict()
                diff = sd[layer_name] - init_param
                times = torch.linalg.norm(diff)/torch.linalg.norm(init_param)
                # print(times)
                if times > epsilon:
                    diff = diff / times * epsilon
                    sd[layer_name] = deepcopy(init_param + diff)
                    cloned_model.load_state_dict(sd)

                with torch.no_grad():
                    total = 0
                    total_loss = 0.0
                    correct = 0
                    for inputs, targets in trainloader:
                        outputs = cloned_model(inputs)
                        total += targets.shape[0]
                        total_loss += criterion(outputs, targets).item() * targets.shape[0]
                        _, predicted = outputs.max(1)
                        correct += predicted.eq(targets).sum().item()  
                    
                    total_loss /= total
                    correct /= total

                if total_loss > max_loss:
                    max_loss = total_loss
                    min_acc = correct
            
            layer_sharpness_dict[layer_name[:-len(".weight")]] = max_loss - origin_loss
            print("{:35}, MRC: {:10.2f}, Dropped Robust Acc: {:10.2f}".format(layer_name[:-len(".weight")], max_loss-origin_loss, (origin_acc-min_acc)*100))

    sorted_layer_sharpness = sorted(layer_sharpness_dict.items(), key=lambda x:x[1])
    for (k, v) in sorted_layer_sharpness:
        print("{:35}, Robust Loss: {:10.2f}".format(k, v))
    
    return sorted_layer_sharpness


def train(model, dataloader, optimizer, criterion, device='cuda'):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(tqdm(dataloader, ncols=100)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item() * targets.size(0)

        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss / total, correct / total * 100


def main(model, trainloader, testloader, layer, lr=0.001, optim='SGDM', lr_scheduler='step', momentum=0.9,
         wd=0.0005, epochs=10, lr_decay_gamma=0.1, model_save_dir='./results', cal_mrc=False,
         attack_method=['fgsm', 'mifgsm', 'pgd'], model_name='ResNet18', dataset='cidfar10', device='cuda'):
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    for path in [model_save_dir]:
        if not os.path.isdir(path):
            os.makedirs(path)

    print('==> Building optimizer and learning rate scheduler...')
    optimizer = create_optimizer(optim, model, lr, momentum, weight_decay=wd)
    # print(optimizer)
    lr_decays = [int(epochs // 2)]
    scheduler = create_scheduler(lr_scheduler, epochs, lr_decay_gamma, optimizer, lr_decays=lr_decays)
    # print(scheduler)

    criterion = nn.CrossEntropyLoss()

    init_sd = deepcopy(model.state_dict())
    torch.save(init_sd, join(model_save_dir, f"{dataset}_{model_name}_init_params.pth"))

    evalulate_robustness = evaluate_cifar_robustness

    if cal_mrc:
        layer_sharpness(deepcopy(model), epsilon=0.1)
        exit()

    assert layer is not None

    for name, param in model.named_parameters():
        param.requires_grad = False
        if layer in name:
            param.requires_grad = True

    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"==> Epoch {epoch} Training... ")
        train_loss, train_acc = train(model, trainloader, optimizer, criterion, device=device)
        print(f"==> Train loss: {train_loss:.2f}, train acc: {train_acc:.2f}%")
        print("==> Testing...")
        test_loss, test_acc = evaluate(model, testloader, criterion)
        print(f"==> Test loss: {test_loss:.2f}, test acc: {test_acc:.2f}%")
        state = {
            'model': model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if test_acc > best_acc:
            best_acc = test_acc
            params = f"{dataset}_{model_name}_best_params.pth"
            print('==> Saving best params...')
            torch.save(state, join(model_save_dir, params))
        scheduler.step()

    checkpoint = torch.load(join(model_save_dir, f"{dataset}_{model_name}_best_params.pth"))
    model.load_state_dict(checkpoint["model"])

    return interpolation(init_sd, deepcopy(model.state_dict()), model, testloader, criterion, model_save_dir,
                         attack_methods=attack_method, device=device, eval_robustness_func=evalulate_robustness)


if __name__ == "__main__":
    trainloader, testloader = create_dataloader(128, dataset='cifar10')
    save_path = join(dirname(__file__), 'results')
    model = create_model('ResNet18', num_classes=10, device='cuda', resume='./ResNet18_CIFAR10.pth')
    main(model=model, trainloader=trainloader, testloader=testloader, layer='layer2.1.conv2', epochs=1, model_save_dir=save_path)






