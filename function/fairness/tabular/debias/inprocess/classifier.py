import torch
import torch.nn as nn
from function.fairness.tabular.fairness_datasets import FairnessDataset
from function.fairness.tabular.models.models import Net
import math
import numpy as np
from sklearn.metrics import confusion_matrix
import copy
# import tensorflow.compat.v1 as tf
import os

class Classifier:
    def __init__(self, input_shape, output_shape=1, sensitive=None, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.num_classes = output_shape
        self.criterion = nn.BCELoss(reduction='none') if output_shape == 1 else nn.CrossEntropyLoss(reduction='none')
        model = self.get_model(input_shape=input_shape)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.sensitive = sensitive # TODO unneccsary here, should be decided in data set
        

    def loss(self, x, y, z=None, w=None):
        if w is None:
            w = torch.ones(x.shape[0]).to(self.device)
        y_hat = self.predict(x, z)
        return self.criterion(y_hat, y)*w, y_hat # return element-wise weighted loss according to sample loss

    def get_model(self, input_shape):
        return Net(input_shape, self.num_classes, 0)
    
    def sample_batch(self, itr, batch_size, X, Y, Z, W):
        x=X[itr*batch_size: (itr+1)*batch_size].float().to(self.device)
        y=Y[itr*batch_size: (itr+1)*batch_size].float().to(self.device)
        z=Z[itr*batch_size: (itr+1)*batch_size].float().to(self.device)
        w=W[itr*batch_size: (itr+1)*batch_size].float().to(self.device) if W is not None else None
        return x, y, z, w

    def check_sens(self, dataset:FairnessDataset):
        available = list(dataset.privileged.keys())
        if self.sensitive is None:
            self.sensitive = available[0]
        if self.sensitive not in available:
            raise ValueError(f"invalid sensitive value: \'{self.sensitive}\' is not in the dataset.")
        idx = available.index(self.sensitive)
        return idx

    def state_dict(self):
        state_dict = {
            'model': self.model.state_dict()
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train(self, dataset: FairnessDataset, epochs=200, batch_size=None, thresh=0.5, save_folder=''):
        idx = self.check_sens(dataset)
        # data prepataion
        train_idx, test_idx, dev_idx = dataset.train_idx, dataset.test_idx, dataset.dev_idx
        X, Y, Z, W = map(lambda x: np.array(x, np.float32),(dataset.X, dataset.Y, dataset.Z, dataset.weights))
        X_train, X_test, X_dev, Y_train, Y_test, Y_dev, Z_train, Z_test, Z_dev, W_train = map(torch.from_numpy, (X[train_idx], X[test_idx], X[dev_idx], Y[train_idx], Y[test_idx], Y[dev_idx], Z[train_idx, idx], Z[test_idx, idx], Z[dev_idx, idx], W[train_idx]))
        # X_train, X_test, Y_train, Y_test, Z_train, Z_test, W_train = map(torch.from_numpy, (X[train_idx], X[test_idx], Y[train_idx], Y[test_idx], Z[train_idx, idx], Z[test_idx, idx], W[train_idx]))
        # X_train, X_test, Y_train, Y_test, Z_train, Z_test = X_train.to(self.device), X_test.to(self.device), Y_train.to(self.device), Y_test.to(self.device), Z_train.to(self.device), Z_test.to(self.device)
        max_acc = 0
        # X_train = X_train.float()
        # X_test = X_test.float()
        # training
        sensattrs = list(dataset.privileged.keys())
        li = dataset.X.columns.tolist()
        for m in sensattrs:
            if m == 'sex':
                sens2 = li.index(m)
            else:
                sens1 = li.index(m)
 
        training_size = len(dataset)
        batch_size = training_size if batch_size is None else batch_size
        itrs = math.ceil(training_size / batch_size)
        print(f"start training with training size : {training_size}, batch size: {batch_size}, max epoch: {epochs}, running device: {self.device}")
        for i in range(epochs):
            for j in range(itrs):
                #state_dict = torch.load(os.path.join(save_folder, 'best.pth'))
                #self.model.load_state_dict(state_dict['model'])
                self.model.train()
                x, y, z, w = self.sample_batch(j, batch_size, X_train, Y_train, Z_train, W_train)
                loss , y_hat = self.loss(x, y, z, w) # 计算损失
                self.optimizer.zero_grad() # 前一步的损失清零
                loss.mean().backward() # 反向传播

                # for p in self.model.parameters():
                #     print(p)

                self.optimizer.step() # 优化
            if (i+1)%100 ==0 : # 这里我们每100次输出相关的信息
                torch.save(self.state_dict(), os.path.join(save_folder, 'ckpt.pth'))
                # 指定模型为计算模式
                self.model.eval()
                x = X_test.float().to(self.device)
                y = Y_test.float().to(self.device)
                z = Z_test.float().to(self.device)
                # y = y.squeeze(dim=1)
                # z = torch.from_numpy(Z_test).long()
                dev_loss = None
                dev_y_hat = None
                with torch.no_grad():
                    # y_hat = self.predict(x, z)
                    dev_loss, dev_y_hat = self.loss(x, y, z, None) # test samples should not be weighted
                    dev_loss = dev_loss.mean().item()
                pred = (dev_y_hat > thresh).float() if self.num_classes == 1 else torch.argmax(dev_y_hat, 1)
                dev_acc = (pred == y).float().mean() * 100
                if dev_acc > max_acc:
                    max_acc = dev_acc
                    yh = dev_y_hat
                    torch.save(self.state_dict(), os.path.join(save_folder, 'best.pth'))
                # z = z.squeeze(dim=1)  
                print(f"Epoch: {i}, Loss: {dev_loss:.6f}, acc: {dev_acc:.2f}%",)

        # 指定模型为计算模式（测试）
        state_dict = None
        if os.path.exists(os.path.join(save_folder, 'best.pth')):
            state_dict = torch.load(os.path.join(save_folder, 'best.pth'))
        elif os.path.exists(os.path.join(save_folder, 'ckpt.pth')):
            state_dict = torch.load(os.path.join(save_folder, 'ckpt.pth'))
        else:
            raise FileNotFoundError("no checkpoints available for testing")

        self.model.load_state_dict(state_dict['model'])
        
        self.model.eval()
        x = X_test.float().to(self.device)
        y = Y_test.float().to(self.device)
        z = Z_test.float().to(self.device)
        loss = None
        y_hat = None
        with torch.no_grad():
            loss, y_hat = self.loss(x, y, z, None) # test samples should not be weighted
            loss = loss.mean().item()
        pred = (y_hat > thresh).float() if self.num_classes == 1 else torch.argmax(y_hat, 1)
        acc = (pred == y).float().mean() * 100

        dim0, dim1 = x.shape
        flag1 = 0
        flag2 = 0
        flag3 = 0
        flag4 = 0
        for i in range(dim0):
            if x[i][sens1] == 0 and x[i][sens2] == 0:
                if flag1 == 0:
                    x1 = x[i].unsqueeze(0)
                    y1 = y[i].unsqueeze(0)
                    z1 = z[i].unsqueeze(0)
                    flag1 = 1
                else:
                    tempx = x[i].unsqueeze(0)
                    tempy = y[i].unsqueeze(0)
                    tempz = z[i].unsqueeze(0)
                    x1 = torch.cat([x1,tempx], dim=0)
                    y1 = torch.cat([y1,tempy], dim=0)
                    z1 = torch.cat([z1,tempz], dim=-1)
            if x[i][sens1] == 0 and x[i][sens2] == 1:
                if flag2 == 0:
                    x2 = x[i].unsqueeze(0)
                    y2 = y[i].unsqueeze(0)
                    z2 = z[i].unsqueeze(0)
                    flag2 = 1
                else:
                    tempx = x[i].unsqueeze(0)
                    tempy = y[i].unsqueeze(0)
                    tempz = z[i].unsqueeze(0)
                    x2 = torch.cat([x2,tempx], dim=0)
                    y2 = torch.cat([y2,tempy], dim=0)
                    z2 = torch.cat([z2,tempz], dim=-1)
            if x[i][sens1] == 1 and x[i][sens2] == 1:
                if flag3 == 0:
                    x3 = x[i].unsqueeze(0)
                    y3 = y[i].unsqueeze(0)
                    z3 = z[i].unsqueeze(0)
                    flag3 = 1
                else:
                    tempx = x[i].unsqueeze(0)
                    tempy = y[i].unsqueeze(0)
                    tempz = z[i].unsqueeze(0)
                    x3 = torch.cat([x3,tempx], dim=0)
                    y3 = torch.cat([y3,tempy], dim=0)
                    z3 = torch.cat([z3,tempz], dim=-1)
            if x[i][sens1] == 1 and x[i][sens2] == 0:
                if flag4 == 0:
                    x4 = x[i].unsqueeze(0)
                    y4 = y[i].unsqueeze(0)
                    z4 = z[i].unsqueeze(0)
                    flag4 = 1
                else:
                    tempx = x[i].unsqueeze(0)
                    tempy = y[i].unsqueeze(0)
                    tempz = z[i].unsqueeze(0)
                    x4 = torch.cat([x4,tempx], dim=0)
                    y4 = torch.cat([y4,tempy], dim=0)
                    z4 = torch.cat([z4,tempz], dim=-1)

        with torch.no_grad():
            loss1, y1_hat = self.loss(x1, y1, z1, None)
            loss2, y2_hat = self.loss(x2, y2, z2, None)
            loss3, y3_hat = self.loss(x3, y3, z3, None)
            loss4, y4_hat = self.loss(x4, y4, z4, None)

        pred1 = (y1_hat > thresh).float() if self.num_classes == 1 else torch.argmax(y1_hat, 1)
        pred2 = (y2_hat > thresh).float() if self.num_classes == 1 else torch.argmax(y2_hat, 1)
        pred3 = (y3_hat > thresh).float() if self.num_classes == 1 else torch.argmax(y3_hat, 1)
        pred4 = (y4_hat > thresh).float() if self.num_classes == 1 else torch.argmax(y4_hat, 1)

        acc1 = ((pred1 == y1).float().mean() * 100).cpu().numpy().astype(np.float64)
        acc2 = ((pred2 == y2).float().mean() * 100).cpu().numpy().astype(np.float64)
        acc3 = ((pred3 == y3).float().mean() * 100).cpu().numpy().astype(np.float64)
        acc4 = ((pred4 == y4).float().mean() * 100).cpu().numpy().astype(np.float64)

        y1_cpu = y1.cpu()
        pred1_cpu = pred1.cpu()
        y2_cpu = y2.cpu()
        pred2_cpu = pred2.cpu()
        y3_cpu = y3.cpu()
        pred3_cpu = pred3.cpu()
        y4_cpu = y4.cpu()
        pred4_cpu = pred4.cpu()
        y1_true = y1_cpu.numpy()
        y1_pred = pred1_cpu.numpy()
        y2_true = y2_cpu.numpy()
        y2_pred = pred2_cpu.numpy()
        y3_true = y3_cpu.numpy()
        y3_pred = pred3_cpu.numpy()
        y4_true = y4_cpu.numpy()
        y4_pred = pred4_cpu.numpy()
        conf_matrix1 = confusion_matrix(y1_true, y1_pred)
        conf_matrix2 = confusion_matrix(y2_true, y2_pred)
        conf_matrix3 = confusion_matrix(y3_true, y3_pred)
        conf_matrix4 = confusion_matrix(y4_true, y4_pred)
        tn1,fp1,fn1,tp1 = conf_matrix1.ravel()
        tn2,fp2,fn2,tp2 = conf_matrix2.ravel()
        tn3,fp3,fn3,tp3 = conf_matrix3.ravel()
        tn4,fp4,fn4,tp4 = conf_matrix4.ravel()
        fpr1 = fp1 / (fp1 + tn1) * 100
        fnr1 = fn1 / (fn1 + tp1) * 100
        tpr1 = tp1 / (tp1 + fn1) * 100
        tnr1 = tn1 / (tn1 + fp1) * 100
        fpr2 = fp2 / (fp2 + tn2) * 100
        fnr2 = fn2 / (fn2 + tp2) * 100
        tpr2 = tp2 / (tp2 + fn2) * 100
        tnr2 = tn2 / (tn2 + fp2) * 100
        fpr3 = fp3 / (fp3 + tn3) * 100
        fnr3 = fn3 / (fn3 + tp3) * 100
        tpr3 = tp3 / (tp3 + fn3) * 100
        tnr3 = tn3 / (tn3 + fp3) * 100
        fpr4 = fp4 / (fp4 + tn4) * 100
        fnr4 = fn4 / (fn4 + tp4) * 100
        tpr4 = tp4 / (tp4 + fn4) * 100
        tnr4 = tn4 / (tn4 + fp4) * 100

        print(f"Loss: {loss:.6f}, acc: {acc:.2f}%",)
        print(f"acc1: {acc1:.2f}%, fpr1: {fpr1:.2f}%, fnr1: {fnr1:.2f}%, tpr1: {tpr1:.2f}%, tnr1: {tnr1:.2f}%", )
        print(f"acc2: {acc2:.2f}%, fpr2: {fpr2:.2f}%, fnr2: {fnr2:.2f}%, tpr2: {tpr2:.2f}%, tnr2: {tnr2:.2f}%", )
        print(f"acc3: {acc3:.2f}%, fpr3: {fpr3:.2f}%, fnr3: {fnr3:.2f}%, tpr3: {tpr3:.2f}%, tnr3: {tnr3:.2f}%", )
        print(f"acc4: {acc4:.2f}%, fpr4: {fpr4:.2f}%, fnr4: {fnr4:.2f}%, tpr4: {tpr4:.2f}%, tnr4: {tnr4:.2f}%", )

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
        x = torch.from_numpy(np.array(dataset.X, np.float32)).float().to(device=self.device)
        z = torch.from_numpy(np.array(dataset.Z)[:, idx]).float().to(device=self.device)
        yh = self.predict(x, z).detach().cpu().numpy()
        predicted_dataset = copy.deepcopy(dataset)
        predicted_dataset.Y = yh
        return predicted_dataset
