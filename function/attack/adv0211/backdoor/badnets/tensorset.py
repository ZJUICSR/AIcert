from torch.utils.data import TensorDataset

class TensorSet(TensorDataset):
    def __init__(self, data, targets, plabel, with_plabel=False, **kwargs):
        super(TensorSet, self).__init__(**kwargs)
        self.data = data
        self.targets = targets
        self.plabel = plabel
        self.with_plabel = with_plabel

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.with_plabel:
            return x, y, self.plabel[index]
        return x, y

    def __len__(self):
        return len(self.data)