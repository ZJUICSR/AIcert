import os
import torch
import glob
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
gtsrb_data_path = os.path.join(os.path.dirname(__file__), 'data', 'gtsrb', 'Train')
MTFL_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'MTFL', 'AFLW')


def get_mnist_data(path=DATA_PATH, number=100, batch_size=1, device='cpu'):
    test_data = datasets.MNIST(path, train=False, download=True, transform=transforms.ToTensor())
    n_class = 10
    image = test_data.data[: number].view(number, 1, 28, 28)
    true_label = test_data.targets[: number]

    image = image.to(torch.float32) / 255.0
    image.to(device)
    true_label.to(device)

    ver_dataset = TensorDataset(torch.tensor(image), torch.tensor(true_label))
    ver_data = DataLoader(ver_dataset, batch_size=batch_size, shuffle=True)

    return ver_data, n_class


def get_cifar_data(path=DATA_PATH, number=100, batch_size=10, device='cpu'):
    test_data = datasets.CIFAR10(path,
                                 train=False,
                                 download=True,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                                    std=[0.2023, 0.1994, 0.2010])]))
    # Choose image from the dataset.
    image = test_data.data[: number]
    image = torch.tensor(image.transpose(0, 3, 1, 2))
    label = torch.tensor(test_data.targets[: number])

    image = image.to(torch.float32) / 255.0
    image.to(device)
    label.to(device)

    ver_dataset = TensorDataset(torch.tensor(image), torch.tensor(label))
    ver_dataloader = DataLoader(ver_dataset, batch_size=batch_size, shuffle=True)

    return ver_dataloader, len(test_data.classes)


def get_gtsrb_data(number=100, batch_size=128, data_path=gtsrb_data_path) -> DataLoader:
    data_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor() # 把0-1变换到(-1,1) image=(image-mean)/std
    ])

    all_data = datasets.ImageFolder(data_path, transform=data_transforms)
    if number is not None:
        all_data, _ = torch.utils.data.random_split(dataset=all_data, lengths=[number, len(all_data) - number])

    data_loader = DataLoader(all_data, batch_size=batch_size, shuffle=True)
    return data_loader, 43


def get_MTFL_label(label_file):
    label_dict = dict()
    with open(label_file, 'r', encoding='utf-8') as file_obj:
        lines = file_obj.readlines()
        for i, line in enumerate(lines):
            line_info = line.split(' ')
            if len(line_info) < 3:
                continue
            pic_name = line_info[0].split('/')[-1] if line_info[0] != '' else line_info[1].split('/')[-1]
            gender = line_info[-4]
            label_dict.update({pic_name: gender})
    return label_dict


def get_data_path(data_path):
    label_file = os.path.join(os.path.dirname(data_path), 'testing.txt')
    train_path = os.path.join(os.path.dirname(data_path), 'AFLW_Train')
    if os.path.exists(train_path):
        return train_path
    from shutil import copyfile
    label_dict = get_MTFL_label(label_file)
    male_path = os.path.join(train_path, 'male')
    female_path = os.path.join(train_path, 'female')
    for path in [train_path, male_path, female_path]:
        os.mkdir(path)
    all_pics = glob.glob(os.path.join(data_path, '*.jpg'))
    for pic in all_pics:
        pic_name = pic.split(os.sep)[-1]
        if label_dict[pic_name] == '1':
            copyfile(pic, os.path.join(male_path, pic_name))
        elif label_dict[pic_name] == '2':
            copyfile(pic, os.path.join(female_path, pic_name))
    return train_path


def get_MTFL_data(number=10, batch_size=1, data_path=MTFL_DATA_PATH):
    file_path = get_data_path(data_path)

    data_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor() # 把0-1变换到(-1,1) image=(image-mean)/std
    ])

    all_data = datasets.ImageFolder(file_path, transform=data_transforms)
    if number is not None:
        all_data, _ = torch.utils.data.random_split(dataset=all_data, lengths=[number, len(all_data) - number])

    data_loader = DataLoader(all_data, batch_size=batch_size, shuffle=True)

    return data_loader, 2


if __name__ == '__main__':
    data = get_cifar_data()
    # for d, l in data:
    #     print(d)
    #     print(l)
    #     break