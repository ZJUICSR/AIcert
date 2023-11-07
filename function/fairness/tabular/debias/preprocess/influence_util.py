# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
from typing import Sequence
import numpy as np
# import gurobipy as gp
from debias.inprocess.influence_classifier import IFClassifier

def lp(fair_infl: Sequence, util_infl: Sequence, fair_loss: float, alpha: float, beta: float,
       gamma: float) -> np.ndarray:
    num_sample = len(fair_infl)
    max_fair = sum([v for v in fair_infl if v < 0.])
    max_util = sum([v for v in util_infl if v < 0.])

    print("Maximum fairness promotion: %.5f; Maximum utility promotion: %.5f;" % (max_fair, max_util))

    all_one = np.array([1. for _ in range(num_sample)])
    fair_infl = np.array(fair_infl)
    util_infl = np.array(util_infl)
    model = gp.Model()
    x = model.addMVar(shape=(num_sample,), lb=0, ub=1)

    if fair_loss >= -max_fair:
        print("=====> Fairness loss exceeds the maximum availability")
        model.addConstr(util_infl @ x <= 0. * max_util, name="utility")
        model.addConstr(all_one @ x <= alpha * num_sample, name="amount")
        model.setObjective(fair_infl @ x)
        model.optimize()
    else:
        model.addConstr(fair_infl @ x <= beta * -fair_loss, name="fair")
        model.addConstr(util_infl @ x <= gamma * max_util, name="util")
        model.setObjective(all_one @ x)
        model.optimize()

    print("Total removal: %.5f; Ratio: %.3f%%\n" % (sum(x.X), (sum(x.X) / num_sample) * 100))

    return 1 - x.X

def grad_ferm(grad_fn: IFClassifier.grad, x: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Fair empirical risk minimization for binary sensitive attribute
    Exp(L|grp_0) - Exp(L|grp_1)
    """

    N = x.shape[0]

    idx_grp_0_y_1 = [i for i in range(N) if s[i] == 0 and y[i] == 1]
    idx_grp_1_y_1 = [i for i in range(N) if s[i] == 1 and y[i] == 1]

    grad_grp_0_y_1, _ = grad_fn(x=x[idx_grp_0_y_1], y=y[idx_grp_0_y_1])
    grad_grp_1_y_1, _ = grad_fn(x=x[idx_grp_1_y_1], y=y[idx_grp_1_y_1])

    return (grad_grp_0_y_1 / len(idx_grp_0_y_1)) - (grad_grp_1_y_1 / len(idx_grp_1_y_1))


def loss_ferm(loss_fn: IFClassifier.loss_np, x: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    N = x.shape[0]

    idx_grp_0_y_1 = [i for i in range(N) if s[i] == 0 and y[i] == 1]
    idx_grp_1_y_1 = [i for i in range(N) if s[i] == 1 and y[i] == 1]

    loss_grp_0_y_1 = loss_fn(x[idx_grp_0_y_1], y[idx_grp_0_y_1])
    loss_grp_1_y_1 = loss_fn(x[idx_grp_1_y_1], y[idx_grp_1_y_1])

    return (loss_grp_0_y_1 / len(idx_grp_0_y_1)) - (loss_grp_1_y_1 / len(idx_grp_1_y_1))


def grad_dp(grad_fn: IFClassifier.grad_pred, x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """ Demographic parity """

    N = x.shape[0]

    idx_grp_0 = [i for i in range(N) if s[i] == 0]
    idx_grp_1 = [i for i in range(N) if s[i] == 1]

    grad_grp_0, _ = grad_fn(x=x[idx_grp_0])
    grad_grp_1, _ = grad_fn(x=x[idx_grp_1])

    return (grad_grp_1 / len(idx_grp_1)) - (grad_grp_0 / len(idx_grp_0))


def loss_dp(x: np.ndarray, s: np.ndarray, pred: np.ndarray) -> float:
    N = x.shape[0]

    idx_grp_0 = [i for i in range(N) if s[i] == 0]
    idx_grp_1 = [i for i in range(N) if s[i] == 1]

    pred_grp_0 = np.sum(pred[idx_grp_0])
    pred_grp_1 = np.sum(pred[idx_grp_1])

    return (pred_grp_1 / len(idx_grp_1)) - (pred_grp_0 / len(idx_grp_0))
