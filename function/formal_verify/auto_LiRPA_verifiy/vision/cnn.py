import torch
from torch import nn
from tqdm import tqdm
from auto_LiRPA_verifiy.vision.data import get_mnist_data
import os


MODEL_SAVE_PATH = os.path.dirname(__file__)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# This simple model comes from https://github.com/locuslab/convex_adversarial
def mnist_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

    return model


def mnist_model_with_different_activate_function():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.Tanh(),
        Flatten(),
        nn.Linear(32 * 7 * 7, 100),
        nn.Sigmoid(),
        nn.Linear(100, 10),
        nn.LeakyReLU()
    )

    return model


def train_single_epoch(detect_model,
                       dataloader: torch.utils.data.DataLoader,
                       lr_scheduler,
                       loss_func,
                       optimizer,
                       epoch: int,
                       device='cuda'):
    detect_model.train()
    lr_scheduler.step(epoch=epoch)
    criterion = loss_func
    train_loss = 0
    train_acc = 0
    for batch_idx, data in tqdm(enumerate(dataloader), ncols=100, desc=f'Epoch {epoch}'):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = detect_model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(output, 1)
        acc = (pred.data == labels.long()).sum()
        train_loss += float(loss)
        train_acc += float(acc)
    torch.cuda.empty_cache()
    print(f'epoch {epoch}, loss={round(train_loss / len(dataloader.sampler), 6)}, '
          f'acc={round(train_acc / len(dataloader.sampler), 6)}')
    return round(train_acc / len(dataloader.sampler), 6), round(train_loss / len(dataloader.sampler), 6)


def train_adv_detect_model(dataloader: torch.utils.data.DataLoader,
                           max_epochs=30,
                           target_acc=0.99,
                           device='cuda'):
    torch.manual_seed(0)
    model_name = os.path.join(MODEL_SAVE_PATH, f'cnn_act_func.pkl')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_epoch = -1
    ori_acc, epoch_acc = 0, 0
    detect_model = mnist_model_with_different_activate_function()
    if os.path.exists(model_name):
        model_info = torch.load(model_name)
        detect_model.load_state_dict(model_info['model'])
        train_epoch = model_info['epoch']
        ori_acc = model_info['acc']
        print(f'load model with epoch {train_epoch}, acc={ori_acc}')
    detect_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(detect_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 70, 150], gamma=0.1)

    for epoch_num in range(max_epochs):
        if epoch_num <= train_epoch:
            continue
        epoch_acc, _ = train_single_epoch(detect_model=detect_model,
                                       dataloader=dataloader,
                                       lr_scheduler=lr_scheduler,
                                       loss_func=criterion,
                                       optimizer=optimizer,
                                       epoch=epoch_num,
                                       device=device)
        if epoch_acc > ori_acc:
            ori_acc = epoch_acc
            save_info = {'model': detect_model.state_dict(),
                         'epoch': epoch_num,
                         'acc': epoch_acc}
            torch.save(save_info, os.path.join(MODEL_SAVE_PATH, f'cnn_act_func.pkl'))
        if epoch_acc >= target_acc:
            break

    return {'acc': ori_acc}


if __name__ == '__main__':
    dataloader, _ = get_mnist_data(number=10000, batch_size=100)
    train_adv_detect_model(dataloader=dataloader, device='cuda', max_epochs=100)

