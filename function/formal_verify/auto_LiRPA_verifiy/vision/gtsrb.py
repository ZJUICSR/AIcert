from torchvision import datasets, transforms
import glob
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from third_party.auto_LiRPA.auto_LiRPA_verifiy.vision.model import get_gtsrb_resnet18


data_path = os.path.join(os.path.dirname(__file__), 'data', 'gtsrb', 'Train')


def load_gtsrb_data(number=None, batch_size=128) -> DataLoader:
    data_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()])

    all_data = datasets.ImageFolder(data_path, transform=data_transforms)
    if number is not None:
        all_data, _ = torch.utils.data.random_split(dataset=all_data, lengths=[number, len(all_data) - number])
        print(all_data)
    data_loader = DataLoader(all_data, batch_size=batch_size, shuffle=True)
    return data_loader



def train(model, train_loader, val_loader, epoches, save_name, use_gpu=False):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    for epoch in range(epoches):
        model.train()
        correct = 0
        training_loss = 0
        for data, target in tqdm(train_loader, desc=f'train modal @epoch {epoch + 1}', ncols=80):
            data, target = Variable(data), Variable(target)
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            max_index = output.max(dim=1)[1]
            correct += (max_index == target).sum()
            training_loss += loss
        print(fr'Training set: Average loss: {(training_loss / len(train_loader.dataset)):.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({(100. * correct / len(train_loader.dataset)):.0f})%')

        if val_loader is None:
            save_model(model, epoch, save_name)
            continue

        # 评估模型
        model.eval()
        validation_loss = 0
        correct = 0
        for data, target in tqdm(val_loader, desc=f'var modal @epoch {epoch + 1}', ncols=80):
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
                if use_gpu:
                    data = data.cuda()
                    target = target.cuda()
                output = model(data)
                validation_loss += F.nll_loss(output, target, size_average=False).data.item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= len(val_loader.dataset)
        scheduler.step(np.around(validation_loss, 2))
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))

        save_model(model, epoch, save_name)


def save_model(model, epoch, path):
    check_point = {"epochs": epoch, "model": model.state_dict()}
    torch.save(check_point, path)


def load_model(path):
    model = get_gtsrb_resnet18()
    epochs = 0
    if os.path.exists(path):
        check_point = torch.load(path)
        model.load_state_dict(check_point['model'])
        epochs = check_point['epochs']
    return model, epochs

def main():
    dataloader = load_gtsrb_data()
    model = get_gtsrb_resnet18(in_planes=25)
    # data = torch.randn(10, 3, 32, 32)
    # print(model(data).shape)
    train(model=model, train_loader=dataloader, val_loader=None, save_name=f'model_sss_in{25}.pth', epoches=30)
    # save_model(model, 10, )


if __name__ == '__main__':
    main()

