import torch

def disparate_impact(y_hat, z, thd=0.5, flip=False):
    # threshold
    y_hat = (y_hat >= thd).float()
    # flip if 0 is positive
    if flip:
        y_hat = (y_hat != 1.0).float()
        y = (y != 1.0).float()
    yh_z_0 = y_hat[z == 0.0]
    yh_z_1 = y_hat[z == 1.0]
    yh_1_z_0 = float(torch.sum(yh_z_0 == 1.0)) / int(torch.sum(yh_z_0))
    yh_1_z_1 = float(torch.sum(yh_z_1 == 1.0)) / int(torch.sum(yh_z_1))
    return yh_1_z_0 / yh_1_z_1

def demographic_parity(y_hat, z, thd=0.5, flip=False):
    # threshold
    y_hat = (y_hat >= thd).float()
    # flip if 0 is positive
    if flip:
        y_hat = (y_hat != 1.0).float()
        y = (y != 1.0).float()
    yh_z_0 = y_hat[z == 0.0]
    yh_z_1 = y_hat[z == 1.0]
    yh_1_z_0 = float(torch.sum(yh_z_0 == 1.0)) / int(torch.sum(yh_z_0))
    yh_1_z_1 = float(torch.sum(yh_z_1 == 1.0)) / int(torch.sum(yh_z_1))
    return abs(yh_1_z_0 - yh_1_z_1)


def overall_misc(y_hat, y, thd=0.5):
    # threshold
    y_hat = (y_hat >= thd).float()
    total = y.size()[0]
    misclass = int(torch.sum(y != y_hat))
    return float(misclass) / total

def false_positive(y_hat, y, thd=0.5):
    # threshold
    y_hat = (y_hat >= thd).float()
    y_0 = int(torch.sum(y == 0.0))
    y_0_yh_1 = int(torch.sum((y == 0.0) & (y_hat == 1.0)))
    return float(y_0_yh_1) / y_0

def true_positive(y_hat, y, thd=0.5):
    return false_positive(y_hat, (y != 1.0).float(), thd)

def false_negtive(y_hat, y, thd=0.5):
    return false_positive((y_hat < thd).float(), (y != 1.0).float(), thd)

def true_negtive(y_hat, y, thd=0.5):
    return false_positive((y_hat < thd).float(), y, thd)

def false_omission(y_hat, y, thd=0.5):
    yh_0 = int(torch.sum(y_hat < thd))
    y_1_yh_0 = int(torch.sum((y == 1.0) & (y_hat < thd)))
    return float(y_1_yh_0) / yh_0

def false_discovery(y_hat, y, thd=0.5):
    return false_omission((y_hat < thd).float(), (y != 1.0).float(), thd)

def predictive_equality(y_hat, y, z, thd=0.5, flip=False):
    # flip if 0 is positive
    if flip:
        y_hat = (y_hat < thd).float()
        y = (y != 1.0).float()
    y_z_0 = y[z == 0.0]
    yh_z_0 = y_hat[z == 0.0]
    y_z_1 = y[z == 1.0]
    yh_z_1 = y_hat[z == 1.0]    
    return false_positive(yh_z_1, y_z_1, thd) \
        - false_positive(yh_z_0, y_z_0, thd)

def equal_opportunity(y_hat, y, z, thd=0.5, flip=False):
    return predictive_equality(y_hat, (y != 1.0).float(), z, thd, flip)

def equal_odds(y_hat, y, z, thd=0.5, flip=False):
    return abs(predictive_equality(y_hat, y, z, thd, flip)) \
        + abs(equal_opportunity(y_hat, y, z), thd, flip)

def predictive_parity(y_hat, y, z, thd=0.5, flip=False):
    # flip if 0 is positive
    if flip:
        y_hat = (y_hat < thd).float()
        y = (y != 1.0).float()
    y_z_0 = y[z == 0.0]
    yh_z_0 = y_hat[z == 0.0]
    y_z_1 = y[z == 1.0]
    yh_z_1 = y_hat[z == 1.0]    
    return false_omission((yh_z_1 < thd).float(), y_z_1, thd) \
        - false_omission((yh_z_0 < thd).float(), y_z_0, thd)
        
def group_acc(y_hat, y, z, thd=0.5):
    result = {}
    groups = torch.unique(z)
    for group_val in groups:
        y_z = y[z == group_val]
        yh_z = y_hat[z == group_val]
        acc = 1- overall_misc(yh_z, y_z, thd)
        result[float(group_val)] = acc
        
    t = torch.Tensor([result[k] for k in result.keys()])
    var = float(torch.var(t))
    
    return result, var

# individual fairness
from sklearn.neighbors import NearestNeighbors
import numpy as np
def consistency(x, y, n_neighbors=5):
    r"""Individual fairness metric from [1]_ that measures how similar the
    labels are for similar instances.

    .. math::
        1 - \frac{1}{n\cdot\text{n_neighbors}}\sum_{i=1}^n |\hat{y}_i -
        \sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)} \hat{y}_j|

    Args:
        n_neighbors (int, optional): Number of neighbors for the knn
            computation.

    References:
        .. [1] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,
            "Learning Fair Representations,"
            International Conference on Machine Learning, 2013.
    """

    X = x
    num_samples = X.shape[0]
    y = y

    # learn a KNN on the features
    nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(X)

    # compute consistency score
    consistency = 0.0
    for i in range(num_samples):
        consistency += np.abs(y[i] - np.mean(y[indices[i]]))
    consistency = 1.0 - consistency/num_samples

    return consistency



def main():
    batch_size = 256
    y = (torch.rand(batch_size) > 0.5).int()
    y_hat = (torch.rand(batch_size) > 0.5).int()
    z = (torch.rand(batch_size) > 0.5).int()
    out, var = group_acc(y_hat, y, z)
    # out1 = false_positive(y_hat, y) + false_negtive(y_hat, y)
    # out2 = overall_misc(y_hat, y) * 2
    # print(out1, out2)
    # print(y)
    # print((y!=1.0).float())
    print(out, var)

if __name__ == '__main__':
    main()
    