from fairness_datasets import *

class Preprocess():
    def __init__(self, dataset: FairnessDataset, sensitive=[], single=True, csy=0):
        self.dataset = dataset
        # extract vector of sensitive attribute and label
        self.x, self.z, self.y = np.array(dataset.X), np.array(dataset.Z), np.array(dataset.Y)
        # self.x, self.z, self.y = self.x[dataset.train_idx], self.z[dataset.train_idx], self.y[dataset.train_idx] # only training set are needed for reweighing?
        self.w = dataset.weights

        # mask for privileged an unprivileged
        self.pri_mask = (self.z == 1)
        self.unp_mask = (self.z != 1)

        # split
        if single: # if support only a single sensitive attribute
            if len(sensitive) > 1:
                raise ValueError("Only a single sensitive attirubte is supported.")
            elif len(sensitive) == 0:
                sensitive = [list(dataset.privileged.keys())[csy]]
        # for every sensitive attribute: 1 privileged and 1 unprivileged group for each attribute
        for s in sensitive:
            if s not in list(self.dataset.privileged.keys()):
                raise ValueError("invalid sensitive attribute: should be in the sensitive attribute of dataset")
            self.i = list(self.dataset.privileged.keys()).index(s) # index of the specified sensitive attribute in the list of sensitive attributes

            self.x_z1 = self.x[self.pri_mask[:, self.i]]
            self.x_z0 = self.x[self.unp_mask[:, self.i]]
            self.y_z1 = self.y[self.pri_mask[:, self.i]]
            self.y_z0 = self.y[self.unp_mask[:, self.i]]

    def fit(self):
        # some privous calculation before transformation 
        pass


    def transform(self):
        # return the new fairness-enhanced dataset
        pass
