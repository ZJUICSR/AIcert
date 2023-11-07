import torch
import torch.nn as nn
from fairness_datasets import FairnessDataset
from models.models import Net
import math
import numpy as np
import copy
from . import Classifier

class EnsembleClassifier(Classifier):
    def __init__(self, input_shape, num_partition, portion, sensitive=None, device=None):
        super().__init__(input_shape=input_shape, sensitive=sensitive, device=device)
        self.input_shape = input_shape
        self.num_partition = num_partition
        self.portion = portion
        self.models = [] # base models
        self.idx_group = None # partition of sub training sets

    def get_base_model(self, input_shape=None, sensitive=None):
        return Classifier(input_shape if input_shape else self.input_shape, sensitive if sensitive else self.sensitive)

    def partition(self, train_size, num_partition, portion):
        idx_group = np.random.choice(train_size, size=(num_partition, int(train_size * portion)))
        return idx_group

    def train(self, dataset: FairnessDataset, epochs=200, batch_size=None,thresh=0.5):

        # get sub training set partition
        idx_group = self.partition(len(dataset), self.num_partition, self.portion)

        train_idx, test_idx = dataset.train_idx, dataset.test_idx
        X_test, Z_test, Y_test = torch.from_numpy(np.array(dataset.X[test_idx])).float().to(self.device), torch.from_numpy(np.array(dataset.Z[test_idx])).float().to(self.device), torch.from_numpy(np.array(dataset.Y[test_idx]))
        y_hats = []
        

        for i in range(self.num_partition):
            # get sub training set
            sub_dataset = dataset.subset(idx_group[i], deepcopy=True)
            # get base model
            base_model = self.get_base_model()
            base_model.train(dataset=sub_dataset, epochs=epochs)
            self.models.append(base_model)
            y_hat = base_model.predict(X_test, Z_test)
            y_hat = y_hat.detach().cpu().numpy()
            y_hats.append(y_hat)

        # average confidence: y_hats: num_partitions x num_test_sample x 1score
        pred_confi = np.average(y_hats, axis=0)
        ensembled_label = (pred_confi > thresh).astype(np.float)
        acc = (ensembled_label == Y_test).astype(np.float).mean() * 100
        print(f"Averaged score ensembled test acc: {acc:.2f}")

        # majority vote
        pred_lable = [(y_hats[i] > thresh).astype(np.int) for i in range(self.num_partition)] # num_partitions x num_test_sample
        preds_onehot = np.eye(2)[pred_lable] # num_partitions x num_test_sample x n_classes 
        pred_votes = np.sum(preds_onehot, axis=0) # 
        ensembled_label = np.argmax(pred_votes, axis=-1)
        acc = (ensembled_label == Y_test).astype(np.float).mean() * 100
        print(f"Majority voting ensembled test acc: {acc:.2f}")

        # Y_test = torch.tensor(Y_test).to(self.device)
        # labels1, labels2 = self.predict(X_test, Z_test, thresh=thresh)
        # acc1 = (labels1 == Y_test).float().mean() * 100
        # acc2 = (labels2 == Y_test).float().mean() * 100
        # print(f"Averaged score ensembled test acc: {acc1.item():.2f}")
        # print(f"Majority voting ensembled test acc: {acc2.item():.2f}")

        

    def predict(self, x, z, thresh=0.5):

        y_hats = []
        for i in range(self.num_partition):
            model = self.models[i]
            y_hat = model.predict(x, z)
            y_hats.append(y_hat)
        
        # average confidence
        y_hats = torch.stack(y_hats)
        pred_confi = torch.mean(y_hats, dim=0)
        ensembled_label1 = (pred_confi > thresh).float()
        # return ensemble_label

        # majority vote
        pred_label = (y_hats > thresh).to(torch.int64)
        # pred_lable = [(y_hats[i] > thresh).int() for i in range(self.num_partition)] # num_partitions x num_test_sample
        preds_onehot = torch.nn.functional.one_hot(pred_label, 2) # num_partitions x num_test_sample x n_classes 
        preds_onehot = torch.eye(2).to(self.device)[pred_label, :] # num_partitions x num_test_sample x n_classes 
        pred_votes = torch.sum(preds_onehot, dim=0) # 
        ensembled_label2 = torch.argmax(pred_votes, dim=-1)

        return ensembled_label2

