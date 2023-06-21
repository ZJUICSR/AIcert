import torch
import torch.nn as nn
from fairness_datasets import FairnessDataset
from function.fairness.models.models import Net
import math
import numpy as np
import copy

# domain independent training requires change of the model output 
class Classifier:
    def __init__(self, input_shape, sensitive=None, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model = self.get_model(input_shape=input_shape)
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.sensitive = sensitive

    def loss(self, x, y, z):
        y_hat = self.predict(x, z)
        return self.criterion(y_hat, y), y_hat

    def get_model(self, input_shape):
        return Net(input_shape, 1, 0)

    def train(self, dataset: FairnessDataset, ratio=0.75, epochs=200, batch_size=None,thresh=0.5):
        available = list(dataset.privileged.keys())
        if self.sensitive is None:
            self.sensitive = available[0]
        if self.sensitive not in available:
            raise ValueError(f"invalid sensitive value: \'{self.sensitive}\' is not in the dataset.")
        idx = available.index(self.sensitive)

        # data prepataion
        X_train, X_test, Y_train, Y_test, Z_train, Z_test = dataset.split(ratio)
        X_train, X_test, Y_train, Y_test, Z_train, Z_test = map(torch.from_numpy, (X_train, X_test, Y_train, Y_test, Z_train[:, idx], Z_test[:, idx]))
        # X_train, X_test, Y_train, Y_test, Z_train, Z_test = X_train.to(self.device), X_test.to(self.device), Y_train.to(self.device), Y_test.to(self.device), Z_train.to(self.device), Z_test.to(self.device)
        max_acc = 0
        # X_train = X_train.float()
        # X_test = X_test.float()
        # training
        
        training_size = len(dataset)
        batch_size = training_size if batch_size is None else batch_size
        itrs = math.ceil(training_size / batch_size)
        print(f"start training with training size : {training_size}, batch size: {batch_size}, max epoch: {epochs}, running device: {self.device}")
        for i in range(epochs):
            for j in range(itrs):
                self.model.train()
                x=X_train[j*batch_size: (j+1)*batch_size].float().to(self.device)
                y=Y_train[j*batch_size: (j+1)*batch_size].float().to(self.device)
                z=Z_train[j*batch_size: (j+1)*batch_size].float().to(self.device)
                # y = y.squeeze(dim=1)
                # predict_prob=self.model(x)
                # y_hat = z * predict_prob[:, 0] + (1-z) * predict_prob[:, 1]
                # y_hat = self.predict(x, z)
                loss , y_hat = self.loss(x, y, z) # 计算损失
                self.optimizer.zero_grad() # 前一步的损失清零
                loss.backward() # 反向传播

                # for p in self.model.parameters():
                #     print(p)

                self.optimizer.step() # 优化
            if (i+1)%100 ==0 : # 这里我们每100次输出相关的信息
                # 指定模型为计算模式
                self.model.eval()
                x = X_test.float().to(self.device)
                y = Y_test.float().to(self.device)
                z = Z_test.float().to(self.device)
                # y = y.squeeze(dim=1)
                # z = torch.from_numpy(Z_test).long()
                loss = None
                y_hat = None
                with torch.no_grad():
                    # y_hat = self.predict(x, z)
                    loss, y_hat = self.loss(x, y, z)
                    loss = loss.item()
                pred = (y_hat > thresh).float()
                acc = (pred == y).float().mean() * 100
                if acc > max_acc:
                    max_acc = acc
                    yh = y_hat
                # z = z.squeeze(dim=1)
                print(f"Epoch: {i}, Loss: {loss:.6f}, acc: {acc:.2f}%", )

    def predict(self, x, z):
        self.model.eval()
        yh = self.model(x)
        # yh = torch.unsqueeze(yh,dim=-1)
        return yh

    def predicted_dataset(self, dataset):
        available = list(dataset.privileged.keys())
        if self.sensitive is None:
            self.sensitive = available[0]
        if self.sensitive not in available:
            raise ValueError(f"invalid sensitive value: \'{self.sensitive}\' is not in the dataset.")
        idx = available.index(self.sensitive)
        x = torch.from_numpy(np.array(dataset.X)).float().to(device=self.device)
        z = torch.from_numpy(np.array(dataset.Z)[:, idx]).float().to(device=self.device)
        yh = self.predict(x, z).detach().cpu().numpy()
        predicted_dataset = copy.deepcopy(dataset)
        predicted_dataset.Y = yh
        return predicted_dataset

class DomainIndependentClassifier(Classifier):
    def __init__(self, input_shape, sensitive=None, device=None):
        super().__init__(input_shape, sensitive, device)

    def loss(self, x, y, z):
        predict_prob = self.model(x)
        yh = z * predict_prob[:, 0] + (1-z) * predict_prob[:, 1]
        yh = torch.unsqueeze(yh,dim=-1)
        loss = self.criterion(yh, y)
        return loss, yh

    def get_model(self, input_shape):
        return Net(input_shape, 2, 0) # domain independent classifier requires an prediction for each domain

    def predict(self, x, z):
        self.model.eval()
        predict_prob = self.model(x)
        yh = z * predict_prob[:, 0] + (1-z) * predict_prob[:, 1]
        yh = torch.unsqueeze(yh,dim=-1)
        return yh

class FADClassifier(Classifier):
    def __init__(self, input_shape, sensitive=None, device=None):
        super().__init__(input_shape, sensitive, device)
        self.criterion_bias = nn.CrossEntropyLoss()

    def get_model(self, input_shape):
        return Net(input_shape, 1, 50) # domain independent classifier requires an prediction for each domain

    def loss(self, x, y, z):
        y_hat, z_hat = self.model(x)
        loss = self.criterion(y_hat, y) + self.criterion_bias(z_hat, torch.unsqueeze(z, dim=-1))
        return loss, y_hat

    def predict(self, x, z):
        self.model.eval()
        y_hat, z_hat = self.model(x)
        # y_hat = torch.unsqueeze(y_hat,dim=-1)
        return y_hat