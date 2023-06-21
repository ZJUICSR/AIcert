from fairness_datasets import *
from debias.lfr_helpers import LFR_optim_objective, get_xhat_y_hat
import scipy.optimize as optim
import copy

class Preprocess():
    def __init__(self, dataset: FairnessDataset, sensitive=[], single=True):
        self.dataset = dataset
        # extract vector of sensitive attribute and label
        self.x, self.z, self.y = np.array(dataset.X), np.array(dataset.Z), np.array(dataset.Y)
        self.w = dataset.weights

        # mask for privileged an unprivileged
        self.pri_mask = (self.z == 1)
        self.unp_mask = (self.z != 1)

        # split
        if single: # if support only a single sensitive attribute
            if len(sensitive) > 1:
                raise ValueError("Only a single sensitive attirubte is supported.")
            elif len(sensitive) == 0:
                sensitive = [list(dataset.privileged.keys())[0]]
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

class LFR(Preprocess):
    def __init__(self, dataset: FairnessDataset, sensitive=[],k=5, Ax=0.01, Ay=1.0, Az=50.0, print_interval=250, verbose=0):
        self.k = k # number of prototypes
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        self.print_interval=print_interval
        self.verbose = verbose
        # extract vector of sensitive attribute and label
        super(LFR, self).__init__(dataset, sensitive, True)

    def fit(self, maxiter=5000, maxfun=5000):

        # Initialize the LFR optim objective parameters
        self.num_samples, self.num_features = np.shape(self.x)
        params_init = np.random.uniform(size=self.k + self.num_features * self.k)
        bnd = [(0,1)]*self.k + [(None, None)] * self.num_features*self.k
        LFR_optim_objective.steps = 0

        self.learned_model = optim.fmin_l_bfgs_b(LFR_optim_objective, x0=params_init, epsilon=1e-5, args=(self.x_z0, self.x_z1, self.y_z0[:, 0], self.y_z1[:, 0], self.k, self.Ax, self.Ay, self.Az, self.print_interval, self.verbose), bounds=bnd, approx_grad=True, maxfun=maxfun, maxiter=maxiter, disp=self.verbose)[0]

        self.w = self.learned_model[:self.k]
        self.prototypes = self.learned_model[self.k:].reshape((self.k, self.num_features))

        return self


    def transform(self, threshold=0.5):
        _, xh_z0, yh_z0 = get_xhat_y_hat(self.prototypes, self.w, self.x_z0)
        _, xh_z1, yh_z1 = get_xhat_y_hat(self.prototypes, self.w, self.x_z1)

        xh = np.zeros(shape=np.shape(self.x))
        yh = np.zeros(shape=np.shape(self.y))
        xh[self.unp_mask[:, self.i]] = xh_z0
        xh[self.pri_mask[:, self.i]] = xh_z1
        yh[self.pri_mask[:, self.i]] = np.reshape(yh_z1, [-1, 1])
        yh[self.unp_mask[:, self.i]] = np.reshape(yh_z0, [-1, 1])
        # transform into binary lables
        yh_bin = (np.array(yh) > threshold).astype(np.float64)

        new_dataset = copy.deepcopy(self.dataset)
        new_dataset.X = xh
        new_dataset.Y = yh_bin

        return new_dataset

class Reweighing(Preprocess):
    def __init__(self, dataset: FairnessDataset, sensitive=[]):
        super().__init__(dataset, sensitive, True)

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

        









