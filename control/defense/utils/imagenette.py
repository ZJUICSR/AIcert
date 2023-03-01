import os.path as osp
from collections import defaultdict as dd
import os
import numpy as np
from tqdm import tqdm
from torchvision.datasets.folder import ImageFolder



class ImageNette(ImageFolder):

    def __init__(self, train=True, transform=None, target_transform=None):
        root = osp.join('/mnt/data/models/mnt/data/ywb_bk', 'Dataset', 'imagenette')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, '...'
            ))

        root = osp.join(root, 'train') if train else osp.join(root, 'val')

        # Initialize ImageFolder
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        # Use 'train' folder as train set and 'val' folder as test set

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))


    def load_img_and_labels(self, train='train'):
        import torch
        if os.path.exists(os.path.join(self.root, '../imagenette_img_'+ train+'.pth')):
            imgs = torch.load(os.path.join(self.root, '../imagenette_img_'+ train+'.pth'))
        else:
            imgs = []
            for i in tqdm(range(len(self))):
                imgs.append(self[i][0])
            torch.save(imgs, os.path.join(self.root, '../imagenette_img_'+train+'.pth'))
        return imgs, self.targets