import numpy as np

# util class for flip and threshold operation
def flip_thresh(y_hat, y, thd=0.5, flip=False):
    if thd:
        y_hat = (y_hat >= thd).astype(np.float)
    # flip if 0 is positive
    if thd and flip:
        y_hat = (y_hat != 1.0).astype(np.float)
        y = (y != 1.0).astype(np.float)
    return y_hat, y

# label-agnostic group fairness metrics
def disparate_impact(y_hat, y, z, w, thd=0.5, flip=False):
    y_hat, y = flip_thresh(y_hat=y_hat, y=y, thd=thd, flip=flip)
    yh_z_0 = y_hat[z == 0.0]
    yh_z_1 = y_hat[z == 1.0]
    w_z_0 = w[z == 0.0]
    w_z_1 = w[z == 1.0]
    yh_1_z_0 = float(np.sum(w_z_0*yh_z_0)) / int(np.sum(w_z_0))
    yh_1_z_1 = float(np.sum(w_z_1*yh_z_1)) / int(np.sum(w_z_1))
    return min(yh_1_z_0 / (yh_1_z_1 + 1e-6), yh_1_z_1 / (yh_1_z_0 + 1e-6))


def demographic_parity(y_hat, y, z, w, thd=0.5, flip=False):
    y_hat, y = flip_thresh(y_hat=y_hat, y=y, thd=thd, flip=flip)
    yh_z_0 = y_hat[z == 0.0]
    yh_z_1 = y_hat[z == 1.0]
    w_z_0 = w[z == 0.0]
    w_z_1 = w[z == 1.0]
    yh_1_z_0 = float(np.sum(w_z_0*yh_z_0)) / int(np.sum(w_z_0))
    yh_1_z_1 = float(np.sum(w_z_1*yh_z_1)) / int(np.sum(w_z_1))
    return abs(yh_1_z_0 - yh_1_z_1)


def normalized_DP(y_hat, y, z, w, thd=0.5, flip=False):
    y_hat, y = flip_thresh(y_hat=y_hat, y=y, thd=thd, flip=flip)
    yh_z_0 = y_hat[z == 0.0]
    yh_z_1 = y_hat[z == 1.0]
    w_z_0 = w[z == 0.0]
    w_z_1 = w[z == 1.0]
    yh_1_z_0 = float(np.sum(w_z_0*yh_z_0)) / int(np.sum(w_z_0))
    yh_1_z_1 = float(np.sum(w_z_1*yh_z_1)) / int(np.sum(w_z_1))
    return abs(yh_1_z_0 - yh_1_z_1) / min(yh_1_z_0 / (yh_1_z_1 + 1e-6), yh_1_z_1 / (yh_1_z_0 + 1e-6))


# single group evaluation metric(with out sensitive attribute z)
def overall_misc(y_hat, y, w, thd=0.5):
    y_hat, y = flip_thresh(y_hat=y_hat, y=y, thd=thd)
    total = float(np.sum(w))
    misclass = float(np.sum((w*(1-y_hat))[y != 0]) + np.sum((w*(1-y_hat))[y != 0]))
    return float(misclass) / total

def false_positive_num(y_hat, y, w, thd=0.5):
    y_hat, y = flip_thresh(y_hat=y_hat, y=y, thd=thd)
    y_0_yh_1 = float(np.sum((w*y_hat)[(y == 0.0)]))
    return float(y_0_yh_1)

def true_positive_num(y_hat, y, w, thd=0.5):
    return false_positive_num(y_hat, (y != 1.0).astype(np.float), w, thd)

def false_negtive_num(y_hat, y, w, thd=0.5):
    return false_positive_num((1 - y_hat).astype(np.float), (y != 1.0).astype(np.float), w, thd)

def true_negtive_num(y_hat, y, w, thd=0.5):
    return false_positive_num((1 - y_hat).astype(np.float), y, w, thd)



def false_positive(y_hat, y, w, thd=0.5):
    y_hat, y = flip_thresh(y_hat=y_hat, y=y, thd=thd)
    y_0 = float(np.sum(w[y == 0.0]))
    y_0_yh_1 = float(np.sum((w * y_hat)[(y == 0.0)]))
    return float(y_0_yh_1) / (y_0+1e-6)

def true_positive(y_hat, y, w, thd=0.5):
    return false_positive(y_hat, (y != 1.0).astype(np.float), w, thd)

def false_negtive(y_hat, y, w, thd=0.5):
    return false_positive((1 - y_hat).astype(np.float), (y != 1.0).astype(np.float), w, thd)

def true_negtive(y_hat, y, w, thd=0.5):
    return false_positive((1 - y_hat).astype(np.float), y, w, thd)

def false_omission(y_hat, y, w, thd=0.5):
    y_hat, y = flip_thresh(y_hat=y_hat, y=y, thd=thd)
    yh_0 = float(np.sum(w*(1-y_hat)))
    y_1_yh_0 = int(np.sum((w*(1-y_hat))[y == 1.0]))
    return float(y_1_yh_0) / (yh_0+1e-6)

def false_discovery(y_hat, y, w, thd=0.5):
    return false_omission((1 - y_hat).astype(np.float), (y != 1.0).astype(np.float), w, thd)

def precision(y_hat, y, w, thd=0.5):
    return true_positive(y_hat, y, w, thd=0.5) \
           / (true_positive(y_hat, y, w, thd=0.5) + false_positive(y_hat, y, w, thd=0.5) + 1e-6)

def F1_score(y_hat, y, w, thd=0.5):
    return (2.0 * precision(y_hat, y, w, thd=0.5) * true_positive(y_hat, y, w, thd=0.5)) \
           / (precision(y_hat, y, w, thd=0.5) + true_positive(y_hat, y, w, thd=0.5)+ 1e-6)

# model fairness metrics
def predictive_equality(y_hat, y, z, w, thd=0.5, flip=False):
    y_hat, y = flip_thresh(y_hat=y_hat, y=y, thd=thd, flip=flip)
    y_z_0 = y[z == 0.0]
    yh_z_0 = y_hat[z == 0.0]
    w_z_0 = w[z == 0.0]
    y_z_1 = y[z == 1.0]
    yh_z_1 = y_hat[z == 1.0]    
    w_z_1 = w[z == 1.0]
    return abs(false_positive(yh_z_1, y_z_1, w_z_1, thd) \
        - false_positive(yh_z_0, y_z_0, w_z_0, thd))

def predictive_difference(y_hat, y, z, w, thd=0.5, flip=False):
    y_hat, y = flip_thresh(y_hat=y_hat, y=y, thd=thd, flip=flip)
    y_z_0 = y[z == 0.0]
    yh_z_0 = y_hat[z == 0.0]
    w_z_0 = w[z == 0.0]
    y_z_1 = y[z == 1.0]
    yh_z_1 = y_hat[z == 1.0]
    w_z_1 = w[z == 1.0]
    return abs(false_positive(yh_z_1, y_z_1, w_z_1, thd) \
        - false_positive(yh_z_0, y_z_0, w_z_0, thd)) + \
           abs(false_omission((1 - yh_z_1).astype(np.float), y_z_1, w_z_1, thd) \
               - false_omission((1 - yh_z_0).astype(np.float), y_z_0, w_z_0, thd))


def equal_opportunity(y_hat, y, z, w, thd=0.5, flip=False):
    return predictive_equality(y_hat, (y != 1.0).astype(np.float), z, w, thd, flip)

def equal_odds(y_hat, y, z, w, thd=0.5, flip=False):
    return abs(predictive_equality(y_hat, y, z, w, thd, flip)) \
        + abs(equal_opportunity(y_hat, y, z, w, thd, flip))

def predictive_parity(y_hat, y, z, w, thd=0.5, flip=False):
    y_hat, y = flip_thresh(y_hat=y_hat, y=y, thd=thd, flip=flip)
    y_z_0 = y[z == 0.0]
    yh_z_0 = y_hat[z == 0.0]
    w_z_0 = w[z == 0.0]
    y_z_1 = y[z == 1.0]
    yh_z_1 = y_hat[z == 1.0]
    w_z_1 = w[z == 1.0] 
    return abs(false_omission((1 - yh_z_1).astype(np.float), y_z_1, w_z_1, thd) \
        - false_omission((1 - yh_z_0).astype(np.float), y_z_0, w_z_0, thd))


def treatment_equality(y_hat, y, z, w, thd=0.5, flip=False):
    y_hat, y = flip_thresh(y_hat=y_hat, y=y, thd=thd, flip=flip)
    y_z_0 = y[z == 0.0]
    yh_z_0 = y_hat[z == 0.0]
    w_z_0 = w[z == 0.0]
    y_z_1 = y[z == 1.0]
    yh_z_1 = y_hat[z == 1.0]
    w_z_1 = w[z == 1.0]
    fp_z_1 = false_positive(yh_z_1, y_z_1, w_z_1, thd)
    fp_z_0 = false_positive(yh_z_0, y_z_0, w_z_0, thd)
    fn_z_1 = false_negtive(yh_z_1, y_z_1, w_z_1, thd)
    fn_z_0 = false_negtive(yh_z_0, y_z_0, w_z_0, thd)

    return min((fn_z_1/fp_z_1+1e-6) / (fn_z_0/fp_z_0+1e-6), (fn_z_0/fp_z_0+1e-6)/(fn_z_1/fp_z_1+1e-6))

        
def group_acc(y_hat, y, z, thd=0.5):
    result = {}
    groups = np.unique(z)
    for group_val in groups:
        y_z = y[z == group_val]
        yh_z = y_hat[z == group_val]
        acc = 1- overall_misc(yh_z, y_z, thd)
        result[float(group_val)] = acc
        
    t = np.array([result[k] for k in result.keys()])
    var = float(np.var(t))
    
    return result, var

# individual fairness
from sklearn.neighbors import NearestNeighbors
import numpy as np
def consistency(x, y, n_neighbors=5):

    X = x
    num_samples = X.shape[0]
    y = y

    # learn a KNN on the features
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(X)

    # compute consistency score
    consistency = 0.0
    for i in range(num_samples):
        consistency += np.abs(y[i] - np.mean(y[indices[i]]))
    consistency = 1.0 - consistency/num_samples

    return consistency[0]


# 
def dic_operation(dic1, dic2, func):
    if not isinstance(dic1, dict):
        return func(dic1, dic2)
    result = {}
    for key in dic1:
        result[key] = func(dic1[key], dic2[key])
    return result

def main():
    np.random.seed(2)
    batch_size = 12
    y = (np.random.randn(batch_size) > 0.5).astype(np.int32)
    y_hat = (np.random.randn(batch_size) > 0.5).astype(np.int32)
    w = np.ones_like(y)
    z = (np.random.randn(batch_size) > 0.5).astype(np.int32)
    #out, var = group_acc(y_hat, y, z, w)
    out1 = false_positive(y_hat, y,w) + false_negtive(y_hat, y,w)
    out2 = overall_misc(y_hat, y,w) * 2
    print(out1, out2)
    # print(y)
    # print((y!=1.0).astype(np.float))
    #print(out, var)


if __name__ == '__main__':
    main()
    