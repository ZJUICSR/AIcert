
import random
import torch
from torchvision import datasets
from torch.utils.data import Dataset

class MNIST(Dataset):
    def __init__(self, root, transform, train=True, download=False, with_plabel=False, **kwargs):
        super(MNIST, self).__init__()
        self.dsets = datasets.MNIST(root=root, train=train, download=download)
        self.transform = transform
        self.data = self.dsets.data
        self.targets = self.dsets.targets
        self.with_plabel = with_plabel
        self.kwargs = kwargs

        # poisoned index (0/1 stands for clean/poisoned)
        self.pidx = [0] * len(self.data)
        if len(kwargs) != 0:
            assert "tlabel" in kwargs
            assert "pratio" in kwargs
            assert "bd_transform" in kwargs
            for (i, t) in enumerate(self.targets):
                if (random.random() < kwargs["pratio"] and t != kwargs["tlabel"]) or (kwargs["pratio"] == 1):
                    self.pidx[i] = 1

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        poisoned = 0
        if self.pidx[index] == 1:
            img = self.kwargs["bd_transform"](img)
            target = self.kwargs["tlabel"]
            poisoned = 1

        target = torch.tensor(target)
        if not self.with_plabel:
            return img, target
        return img, target, poisoned

    def __len__(self):
        return len(self.data)