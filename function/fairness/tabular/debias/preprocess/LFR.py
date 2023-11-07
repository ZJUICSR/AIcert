from fairness_datasets import *
from debias.preprocess.lfr_helpers import LFR_optim_objective, get_xhat_y_hat
import scipy.optimize as optim
import copy
from . import Preprocess

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
    
        

        









