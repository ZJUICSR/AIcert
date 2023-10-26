from fairness_datasets import *
from . import Preprocess
from debias.inprocess.influence_classifier import IFClassifier
from debias.preprocess.influence_util import *
# from models.models import 
    
class InfluenceReweighing(Preprocess):
    def __init__(self, dataset: FairnessDataset, sensitive=[], single=True, metric='eop', alpha=None, beta=0.5, gamma=0.2):
        super().__init__(dataset, sensitive, single)
        self.metric = metric
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weights = None
        
    def fit(self):
        model = IFClassifier(self.dataset.num_features)
        
        x_train = self.dataset['x', 'train']
        y_train = self.dataset['y', 'train']
        x_test = self.dataset['x', 'test']
        y_test = self.dataset['y', 'test']
        z_test = self.dataset['z', 'test'][:, self.i]
        # vanilla training
        model.train(self.dataset, epochs=3000)
        if self.metric == "eop":
            ori_fair_loss_val = loss_ferm(model.loss_np, x_test, y_test, z_test)
        elif self.metric == "dp":
            pred_val, _ = model.predict(x_test)
            ori_fair_loss_val = loss_dp(x_test, z_test, pred_val)
        else:
            raise ValueError
        # ori_util_loss_val = model.loss_np(x_test, y_test)
        # compute influence

        train_total_grad, train_indiv_grad = model.grad(x_train, y_train)
        util_loss_total_grad, acc_loss_indiv_grad = model.grad(x_test, y_test)
        if self.metric == "eop":
            fair_loss_total_grad = grad_ferm(model.grad, x_test, y_test, z_test)
        elif self.metric == "dp":
            fair_loss_total_grad = grad_dp(model.grad_pred, x_test, z_test)
        else:
            raise ValueError

        hess = model.hess(x_train, y_train)
        # if self.metric == "eop":
        #     fair_loss_total_grad = grad_ferm(model.grad, x_test, y_test, z_test)
        # elif self.metric == "dp":
        #     fair_loss_total_grad = grad_dp(model.grad_pred, x_test, z_test)
        # else:
        #     raise ValueError
        
        util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)
        fair_grad_hvp = model.get_inv_hvp(hess, fair_loss_total_grad)

        util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
        fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)

        self.weights = lp(fair_pred_infl, util_pred_infl, ori_fair_loss_val, self.alpha, self.beta, self.gamma)

    def transform(self):
        new_dataset = copy.deepcopy(self.dataset)
        new_dataset.weights[new_dataset.train_idx] = self.weights
        return new_dataset
        
    # def fit(self, model_class: )
        
    
        

        









