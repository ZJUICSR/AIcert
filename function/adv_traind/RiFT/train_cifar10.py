# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy
from tqdm import tqdm

from .utils import *
from .dataloader import *
from .model import create_model
from .optimizer import *
from os.path import join, dirname


def train(args, model, dataloader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
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


def train_single_epoch(model,
                       dataloader,
                       lr_scheduler,
                       loss_func,
                       optimizer,
                       epoch: int,
                       device='cuda'):
    model.train()
    criterion = loss_func
    train_loss = list()
    train_acc = list()
    loop = tqdm(enumerate(dataloader), ncols=100, desc=f'Train epoch {epoch}', total=len(dataloader),
                colour='blue')
    for batch_idx, data in loop:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        mini_out = model(images)
        mini_loss = criterion(mini_out, labels.long())
        mini_loss.backward()
        optimizer.step()

        _, pred = torch.max(mini_out, 1)
        acc = (pred.data == labels).float().mean()
        train_loss.append(float(mini_loss))
        train_acc.append(float(acc))

        loop.set_postfix({"Loss": f'{np.array(train_loss).mean():.6f}',
                          "Acc": f'{np.array(train_acc).mean():.6f}'})

    torch.cuda.empty_cache()
    lr_scheduler.step(epoch=epoch)
    return np.array(train_acc).mean(), np.array(train_loss).mean()


def train_model(model, dataloader, train_epoch, model_save_dir, device='cuda', model_name='x.pt'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 70, 150], gamma=0.1)
    best_acc = 0
    for epoch in tqdm(range(train_epoch), ncols=100, desc=f'train'):
        acc, _ = train_single_epoch(model=model,
                                    dataloader=dataloader,
                                    lr_scheduler=lr_scheduler,
                                    loss_func=criterion,
                                    optimizer=optimizer,
                                    epoch=epoch,
                                    device=device)
        if acc > best_acc:
            print('==> Saving checkpoints...')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': 5,
            }
        torch.save(state, join(model_save_dir, model_name))

        if acc > 0.99:
            break
    return acc


def main(model, trainloader, testloader, lr=0.001, optim='SGDM', lr_scheduler='step', momentum=0.9,
         wd=0.0005, epochs=50, lr_decay_gamma=0.1, model_save_dir='./results', device='cuda', model_name='x.pth',
         attack_methods=['fgsm', 'mifgsm', 'pgd']):
    seed = 0

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for path in [model_save_dir]:
        if not os.path.isdir(path):
            os.makedirs(path)

    print('==> Building optimizer and learning rate scheduler...')
    optimizer = create_optimizer(optim, net=model, lr=lr, momentum=momentum, weight_decay=wd)
    # print(optimizer)
    lr_decays = [int(epochs // 2)]
    scheduler = create_scheduler(lr_scheduler, epochs,  lr_decay_gamma, optimizer, lr_decays=lr_decays)
    # print(scheduler)

    criterion = nn.CrossEntropyLoss()

    train_model(model=model, dataloader=trainloader, train_epoch=epochs,
                model_save_dir=model_save_dir, device=device, model_name=model_name)
    _, train_acc = evaluate(model, trainloader, criterion)
    _, test_acc = evaluate(model, testloader, criterion)
    robust_acc = evaluate_cifar_robustness(model, dataloader=testloader, attack_methods=attack_methods, device=device)

    print(f'==> train_acc={train_acc}, test_acc={test_acc}, robust_acc={robust_acc}')

    print('==> Saving checkpoints...')
    state = {
        'model': model.state_dict(),
        'acc': test_acc,
        'epoch': 5,
    }
    torch.save(state, join(model_save_dir, model_name))

    return {"model": model, 'evaluate': {'test_acc': test_acc, 'robust_acc': robust_acc}}


if __name__ == "__main__":
    trainloader, testloader = create_dataloader(128, dataset='cifar10')
    save_path = join(dirname(__file__), 'results')
    model = create_model(model_name='ResNet18', num_classes=10, device='cuda', resume=None)
    results = main(model=model, trainloader=trainloader, testloader=testloader, model_save_dir=save_path, epochs=1)
    print(results)





