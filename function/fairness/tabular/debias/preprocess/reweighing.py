from fairness_datasets import *
import copy
from . import Preprocess

class Reweighing(Preprocess):
    def __init__(self, dataset: FairnessDataset, sensitive=[], csy=0):
        super().__init__(dataset, sensitive, True, csy=csy)

        # initialize weight for all y and z
        self.w_y1_z1 = 1
        self.w_y0_z1 = 1
        self.w_y1_z0 = 1
        self.w_y0_z0 = 1

    def fit(self):
        self.fav_mask = np.squeeze(self.y == 1, axis=-1)
        self.unf_mask = np.squeeze(self.y != 1, axis=-1)
        n = np.sum(self.w, dtype=np.float64)
        n_z1 = np.sum(self.w[self.pri_mask[:, self.i]], dtype=np.float64)
        n_z0 = np.sum(self.w[self.unp_mask[:, self.i]], dtype=np.float64)
        n_y1 = np.sum(self.w[self.fav_mask], dtype=np.float64)
        n_y0 = np.sum(self.w[self.unf_mask], dtype=np.float64)

        n_y0_z0 = np.sum(self.w[self.unp_mask[:, self.i] & self.unf_mask], dtype=np.float64)
        n_y1_z0 = np.sum(self.w[self.unp_mask[:, self.i] & self.fav_mask], dtype=np.float64)
        n_y0_z1 = np.sum(self.w[self.pri_mask[:, self.i] & self.unf_mask], dtype=np.float64)
        n_y1_z1 = np.sum(self.w[self.pri_mask[:, self.i] & self.fav_mask], dtype=np.float64)

        self.w_y1_z1 = n_y1*n_z1 / (n * n_y1_z1)
        self.w_y0_z1 = n_y0*n_z1 / (n * n_y0_z1)
        self.w_y1_z0 = n_y1*n_z0 / (n * n_y1_z0)
        self.w_y0_z0 = n_y0*n_z0 / (n * n_y0_z0)

        return self
    
    def transform(self):
        new_dataset = copy.deepcopy(self.dataset)

        # apply reweighing
        new_dataset.weights[self.unp_mask[:, self.i] & self.unf_mask] = self.w_y0_z0
        new_dataset.weights[self.unp_mask[:, self.i] & self.fav_mask] = self.w_y1_z0
        new_dataset.weights[self.pri_mask[:, self.i] & self.unf_mask] = self.w_y0_z1
        new_dataset.weights[self.pri_mask[:, self.i] & self.fav_mask] = self.w_y1_z1

        return new_dataset

    def test(self):
        self.w_y1_z11 = 1
        self.w_y1_z10 = 1
        self.w_y1_z01 = 1
        self.w_y1_z00 = 1
        self.w_y0_z11 = 1
        self.w_y0_z10 = 1
        self.w_y0_z01 = 1
        self.w_y0_z00 = 1

        self.fav_mask = np.squeeze(self.y == 1, axis=-1)
        self.unf_mask = np.squeeze(self.y != 1, axis=-1)
        n = np.sum(self.w, dtype=np.float64)
        n_y1 = np.sum(self.w[self.fav_mask], dtype=np.float64)
        n_y0 = np.sum(self.w[self.unf_mask], dtype=np.float64)
        n_z11 = np.sum(self.w[self.pri_mask[:, 0] & self.pri_mask[:, 1]], dtype=np.float64)
        n_z10 = np.sum(self.w[self.pri_mask[:, 0] & self.unp_mask[:, 1]], dtype=np.float64)
        n_z01 = np.sum(self.w[self.unp_mask[:, 0] & self.pri_mask[:, 1]], dtype=np.float64)
        n_z00 = np.sum(self.w[self.unp_mask[:, 0] & self.unp_mask[:, 1]], dtype=np.float64)

        n_y1_z11 = np.sum(self.w[self.pri_mask[:, 0] & self.pri_mask[:, 1] & self.fav_mask], dtype=np.float64)
        n_y1_z10 = np.sum(self.w[self.pri_mask[:, 0] & self.unp_mask[:, 1] & self.fav_mask], dtype=np.float64)
        n_y1_z01 = np.sum(self.w[self.unp_mask[:, 0] & self.pri_mask[:, 1] & self.fav_mask], dtype=np.float64)
        n_y1_z00 = np.sum(self.w[self.unp_mask[:, 0] & self.unp_mask[:, 1] & self.fav_mask], dtype=np.float64)
        n_y0_z11 = np.sum(self.w[self.pri_mask[:, 0] & self.pri_mask[:, 1] & self.unf_mask], dtype=np.float64)
        n_y0_z10 = np.sum(self.w[self.pri_mask[:, 0] & self.unp_mask[:, 1] & self.unf_mask], dtype=np.float64)
        n_y0_z01 = np.sum(self.w[self.unp_mask[:, 0] & self.pri_mask[:, 1] & self.unf_mask], dtype=np.float64)
        n_y0_z00 = np.sum(self.w[self.unp_mask[:, 0] & self.unp_mask[:, 1] & self.unf_mask], dtype=np.float64)

        self.w_y1_z11 = (n_y1 * n_z11) / (n * n_y1_z11)
        self.w_y1_z10 = (n_y1 * n_z10) / (n * n_y1_z10)
        self.w_y1_z01 = (n_y1 * n_z01) / (n * n_y1_z01)
        self.w_y1_z00 = (n_y1 * n_z00) / (n * n_y1_z00)
        self.w_y0_z11 = (n_y0 * n_z11) / (n * n_y0_z11)
        self.w_y0_z10 = (n_y0 * n_z10) / (n * n_y0_z10)
        self.w_y0_z01 = (n_y0 * n_z01) / (n * n_y0_z01)
        self.w_y0_z00 = (n_y0 * n_z00) / (n * n_y0_z00)

        new_dataset = copy.deepcopy(self.dataset)

        new_dataset.weights[self.unp_mask[:, 0] & self.unp_mask[:, 1] & self.unf_mask] = self.w_y0_z00
        new_dataset.weights[self.unp_mask[:, 0] & self.pri_mask[:, 1] & self.unf_mask] = self.w_y0_z01
        new_dataset.weights[self.pri_mask[:, 0] & self.unp_mask[:, 1] & self.unf_mask] = self.w_y0_z10
        new_dataset.weights[self.pri_mask[:, 0] & self.pri_mask[:, 1] & self.unf_mask] = self.w_y0_z11
        new_dataset.weights[self.unp_mask[:, 0] & self.unp_mask[:, 1] & self.fav_mask] = self.w_y1_z00
        new_dataset.weights[self.unp_mask[:, 0] & self.pri_mask[:, 1] & self.fav_mask] = self.w_y1_z01
        new_dataset.weights[self.pri_mask[:, 0] & self.unp_mask[:, 1] & self.fav_mask] = self.w_y1_z10
        new_dataset.weights[self.pri_mask[:, 0] & self.pri_mask[:, 1] & self.fav_mask] = self.w_y1_z11

        return new_dataset
        
    
        

        









