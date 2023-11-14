# -*- coding: utf-8 -*-
import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class AdversarialDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-2].split(".")[0]
        label = 1 if label == "adv" else 0

        return img_transformed, label


def get_adv_dataloader(batch_size=128):
    seed = 0

    original_pic_dir = r'data/002/ori'
    adv_pic_dir = r'data/002/adv'

    ori_list = glob.glob(os.path.join(original_pic_dir, '*.jpg'))
    adv_list = glob.glob(os.path.join(adv_pic_dir, '*.jpg'))

    pic_list = ori_list + adv_list
    print(f"pic_list: {len(pic_list)}")

    adv_labels = [path.split('/')[-2].split('.')[0] for path in pic_list]

    adv_train_list, adv_valid_list = train_test_split(pic_list,
                                                      test_size=0.4,
                                                      stratify=adv_labels,
                                                      random_state=seed)
    adv_valid_label = [path.split('/')[-2].split('.')[0] for path in adv_valid_list]
    adv_valid_list, adv_test_list = train_test_split(adv_valid_list,
                                                     test_size=0.5,
                                                     stratify=adv_valid_label,
                                                     random_state=seed)

    print(f"Train Data: {len(adv_train_list)}")
    print(f"Validation Data: {len(adv_valid_list)}")
    print(f"Test Data: {len(adv_test_list)}")
    # print(train_list[0])

    adv_train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    adv_val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    adv_test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    adv_train_data = AdversarialDataset(adv_train_list, transform=adv_train_transforms)
    adv_valid_data = AdversarialDataset(adv_valid_list, transform=adv_val_transforms)
    adv_test_data = AdversarialDataset(adv_test_list, transform=adv_test_transforms)

    adv_train_loader = DataLoader(dataset=adv_train_data, batch_size=batch_size, shuffle=True)
    adv_valid_loader = DataLoader(dataset=adv_valid_data, batch_size=batch_size, shuffle=True)
    adv_test_loader = DataLoader(dataset=adv_test_data, batch_size=batch_size, shuffle=True)
    return adv_train_loader, adv_valid_loader, adv_test_loader


class CombineOriAdvDataset(Dataset):
    def __init__(self, ori_dataset, adv_dataset):
        self.ori_dataset = ori_dataset
        self.adv_dataset = adv_dataset

    def __len__(self):
        return len(self.ori_dataset) + len(self.adv_dataset)

    def __getitem__(self, idx):
        if idx < len(self.ori_dataset):
            return self.ori_dataset[idx][0], 0
        return self.adv_dataset[idx - len(self.ori_dataset)][0], 1


def combine_ori_adv_dataloader(ori_dataloader: DataLoader,
                               adv_dataloader: DataLoader,
                               batch_size=128,
                               shuffle=True) -> DataLoader:
    '''
    :param ori_dataloader:
    :param adv_dataloader:
    :param batch_size:
    :param shuffle:
    :return:
    '''
    dataset = CombineOriAdvDataset(ori_dataloader.dataset, adv_dataloader.dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



if __name__ == '__main__':
    from torch.utils.data import ConcatDataset
    train_loader, valid_loader, _ = get_adv_dataloader()
    print(dir(train_loader))
    b = {'a': train_loader, 'b': valid_loader}

    dataset = ConcatDataset([train_loader.dataset, valid_loader.dataset])
    print(len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    print(len(dataloader.dataset))

