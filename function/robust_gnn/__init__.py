#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = ''
__copyright__ = 'Copyright © 2021/06/14, NWPU'


import os
from os.path import dirname, join
import json
import torch
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from collections import Counter
from .robust_gcn import RobustGCNModel, train, certify, sparse_tensor
from .utils import train_val_test_split_tabular, load_npz
from .path import RESULT_ROOT, RGCN_ROOT as ROOT

if not os.path.exists(RESULT_ROOT):
    os.makedirs(RESULT_ROOT)


def run_robust_gcn(dataset_name, device=torch.device("cuda:0"), path=RESULT_ROOT, logger_func=print, **kwargs):
    training = RobustGCNTrain(params=kwargs, device=device, path=path)
    return training(dataset_name=dataset_name, logger_func=logger_func)


class RobustGCNTrain(object):
    def __init__(self, params=None, device=torch.device("cuda:0"), path=None):
        if params is None:
            with open(join(ROOT, "params.json"), "r") as f:
                self.params = json.load(f)
        else:
            self.params = params
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)
        valid_params = 'train_Q batch_size q_ratio train_size val_size test_size '\
                       'random_state n_iters burn_in margin_iters'.split()
        for valid_param in valid_params:
            assert valid_param in self.params, f"{valid_param} not in params."
        for param in self.params:
            assert param in valid_params, f"{param} is not required.."
        self.device = device

    def __call__(self, dataset_name, logger_func=print):
        pr = self.params
        # Data loading
        path = self.path
        A, X, z = load_npz(join(dirname(__file__), "data", f"{dataset_name}.npz"))

        def _save_json_dataset(adj_matrix, labels):
            """Figure 1"""
            adj_matrix = coo_matrix(adj_matrix > 0)
            g = nx.Graph()

            for i, j, d in zip(adj_matrix.row, adj_matrix.col, adj_matrix.data):
                g.add_edge(i, j)

            degree = list(d for node, d in g.degree())
            scatters = [(d, int(zv), cnt) for (d, zv), cnt in Counter(zip(degree, list(labels))).items()]
            # pwd = os.path.dirname(os.path.abspath(__file__))
            # pwd1 = os.path.split(pwd)
            # print(pwd1)
            os.chdir(path)
            RGCN_PATH = join(dirname(__file__), path)
            with open(join(RGCN_PATH, 'dataset_figure.json'), 'w') as f:
                json.dump(dict(
                    example_url="https://echarts.apache.org/examples/zh/editor.html?c=scatter-punchCard",
                    desc="Each tuple: degree(x), label(y), count(scatter size)",
                    data=scatters
                ), f, indent=4)
        _save_json_dataset(A, z)

        A = (A + A.T > 0).astype("float32")  # make undirected
        X = (X > 0).astype("float32")  # binarize node attributes
        K = z.max()+1
        N, D = X.shape

        X_t = sparse_tensor(X).cuda()
        y_t = torch.tensor(z.astype("int64"), device="cuda")

        # Hyperparameters
        hidden_sizes = [300, K] if dataset_name == 'citeseer' else [32, K]
        q = int(float(pr["q_ratio"]) * D)
        Q = int(pr['train_Q'])
        batch_size = int(pr['batch_size'])
        burn_in = pr['burn_in']
        margin_iters=int(pr['margin_iters'])
        n_iters=int(pr['n_iters'])

        # Data splitting
        idx_train, idx_val, idx_test = train_val_test_split_tabular(
            np.arange(N), train_size=pr['train_size'], val_size=pr['val_size'], test_size=pr['test_size'],
            stratify=z, random_state=int(pr['random_state']))
        idx_unlabeled = np.union1d(idx_val, idx_test)

        # Model creation
        model_normal = RobustGCNModel(A, [D]+hidden_sizes).cuda()
        model_normal.train()
        model_normal.to(self.device)
        model = RobustGCNModel(A, [D]+hidden_sizes).cuda()
        # print('**********************************************8')
        # print("RobustGCNModel have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
        # print('**********************************************8')
        model.train()
        model.to(self.device)
        torch.cuda.empty_cache()

        # Train normal
        robust_test_result = dict(normal_method=0, robust_method=0)
        train(model_normal, X_t.cuda(), y_t, idx_train, idx_unlabeled, q=q,
              Q=Q, method="Normal", batch_size=batch_size, n_iters=250)
        predictions_test = model_normal.predict(X_t.to_dense().cuda(), idx_test)
        test_accuracy = (predictions_test == y_t[idx_test].cpu()).float().mean().numpy()
        logger_func(f"{np.round(test_accuracy*100,2)}% of test nodes are correctly classified with normal training.")
        torch.cuda.empty_cache()
        robust_at_Q = certify(model_normal, X.astype("float32"), q, 
                              optimize_omega=False, Q=Q,
                              batch_size=batch_size,
                              certify_nonrobustness=False,
                              progress=True)[0].mean()
        logger_func(f"{np.round(robust_at_Q * 100, 2)}% of nodes are certifiably robust "
                    f"for {Q} perturbations after normal training.")
        robust_test_result['normal_method'] = robust_at_Q * 100
        
        certifiable_nodes_normal = []

        Q_range = [1, 5, 10, 12, 15, 25, 35, 50]
        for _Q in Q_range:
            logger_func(f'[Normal] Current: {_Q}')
            certifiable_nodes_normal.append(certify(model_normal, X.astype("float32"), q, 
                                            optimize_omega=False, 
                                            Q=int(_Q), batch_size=batch_size, 
                                            certify_nonrobustness=True,
                                            progress=True))
        robust_normal = np.array([x[0] for x in certifiable_nodes_normal])
        nonrobust_normal = np.array([x[1] for x in certifiable_nodes_normal])

        # Train robust

        train(model, X_t.cuda(), y_t, idx_train, idx_unlabeled, q=q, Q=Q, method="Robust Hinge U",
              burn_in=burn_in, batch_size=batch_size, n_iters=n_iters,
              margin_iters=margin_iters)
        predictions_test = model.predict(X_t.to_dense().cuda(), idx_test)
        test_accuracy = (predictions_test == y_t[idx_test].cpu()).float().mean().numpy()
        logger_func(f"{np.round(test_accuracy*100,2)}% of test nodes are correctly classified after robust training.")

        robust_at_Q = certify(model, X.astype("float32"), q, 
                              optimize_omega=False, Q=Q,
                              batch_size=batch_size,
                              certify_nonrobustness=False,
                              progress=True)[0].mean()
        logger_func(f"{np.round(robust_at_Q * 100, 2)}% of nodes are certifiably robust for {Q} perturbations after robust training.")
        robust_test_result['robust_method'] = robust_at_Q * 100

        certifiable_nodes = []

        Q_range = [1, 5, 10, 12, 15, 25, 35, 50]
        for _Q in Q_range:
            logger_func(f'[Robust] Current: {_Q}')
            certifiable_nodes.append(certify(model, X.astype("float32"), q, 
                                             optimize_omega=False,
                                             Q=int(_Q), batch_size=batch_size,
                                             certify_nonrobustness=True,
                                             progress=True))
        
        robust = np.array([x[0] for x in certifiable_nodes])
        nonrobust = np.array([x[1] for x in certifiable_nodes])
        os.chdir(path)
        RGCN_PATH = join(dirname(__file__), path)
        # Save model
        torch.save(model, join(RGCN_PATH, 'model.pkl'))
        torch.save(model_normal, join(RGCN_PATH, 'model_normal.pkl'))
        self.model = model
        self.model_normal = model_normal

        def _save_json_result():
            std_train_line1 = list(zip(Q_range, 100*robust_normal.mean(1)))
            std_train_line2 = list(zip(Q_range, 100*(1-nonrobust_normal.mean(1))))

            robust_line = list(zip(Q_range, 100*(robust.mean(1))))
            robust_area = []
            for Q_val, val in robust_line:
                robust_area.append([Q_val, val])
            robust_area += [[Q_range[-1], 0], [Q_range[0], 0]]

            nonrobust_line = list(zip(Q_range, 100*(1-nonrobust.mean(1))))
            nonrobust_area = []
            for Q_val, val in nonrobust_line:
                nonrobust_area.append([Q_val, val])
            nonrobust_area += [[Q_range[-1], 100], [Q_range[0], 100]]
            os.chdir(path)
            RGCN_PATH = join(dirname(__file__), path)
            with open(join(RGCN_PATH, 'result.json'), 'w') as f:
                json.dump(dict(
                    example_url=[
                        "https://echarts.apache.org/examples/zh/editor.html?c=custom-cartesian-polygon",
                        "https://echarts.apache.org/examples/zh/editor.html?c=gauge-ring"
                    ],
                    desc="robust_test_result: passed ratio of 2 methods.\nEach part of line_and_area: pair of points",
                    data=dict(
                        robust_test_result=robust_test_result,
                        line_and_area=dict(
                            std_train_line1=list(map(list, std_train_line1)), 
                            std_train_line2=list(map(list, std_train_line2)),
                            robust_line=list(map(list, robust_line)), 
                            robust_area=robust_area, 
                            nonrobust_line=list(map(list, nonrobust_line)), 
                            nonrobust_area=nonrobust_area
                    ))
                ), f, indent=4)
        _save_json_result()

        def _load_json_file(file):
            json_info = dict()
            with open(file, 'r', encoding='utf-8') as f:
                json_info = json.load(f)
            return json_info

        return _load_json_file(join(RGCN_PATH, 'result.json'))


if __name__ == '__main__':
    print("start")

