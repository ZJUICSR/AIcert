import torch
from fairness_datasets import CompasDataset, AdultDataset, GermanDataset
from metrics.dataset_metric import DatasetMetrics
from debias.preprocess import *
# from models.models import LR, Net
from torch import nn, threshold
import random
import numpy as np
from metrics.model_metrics import ModelMetrics
import copy
from debias.inprocess import *
from debias.postprocess import *

seed = 11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# dataset = CompasDataset(features_to_drop=['c_charge_desc'])
# dataset = AdultDataset()
favorable='score_text'
# dataset = AdultDataset(favorable=favorable)
dataset = CompasDataset(favorable=favorable)
# dataset = GermanDataset(favorable=favorable)

# test dataset metrics
metrics = DatasetMetrics(dataset)
print("evaluating fairness of dataset: Compas, target: {} ....".format(favorable))
print("overall favorable rate: ", metrics.favorable_rate())
print("favorable difference: ", metrics.favorable_diff())
print("favorable ratio: ", metrics.favorable_ratio())
print("individual consistency: ", metrics.consistency())
# exit()
# improve dataset fairness
print("improving fairness of dataset using technique: Reweighing ....")
preprocess = Reweighing(dataset)
preprocess.fit()
new_dataset = preprocess.transform()

print("evaluating fairness of fairness-improved dataset ....")
metrics = DatasetMetrics(new_dataset)
print("testing dataset metrics.")
print("overall favorable rate: ", metrics.favorable_rate())
print("favorable difference: ", metrics.favorable_diff())
print("favorable ratio: ", metrics.favorable_ratio())
# print("individual consistency: ", metrics.consistency())
exit()

# test inprocess
print(f"training with number of features: {dataset.num_features}")
# classifer = Classifier(input_shape=dataset.num_features, device="cuda")
# classifer = DomainIndependentClassifier(input_shape=dataset.num_features, device="cuda")
classifer = FADClassifier(input_shape=dataset.num_features, device="cuda")
classifer.train(dataset=dataset, epochs=2000)

predicted_dataset = classifer.predicted_dataset(dataset)
print("evaluating fairness of classifier ....")
model_metrics = ModelMetrics(new_dataset, predicted_dataset)
for metric in ModelMetrics.FAIRNESS_METRICS:
    m = model_metrics.group_fairness_metrics(metric)
    print(metric, ": ", m)


# postprocess = RejectOptionClassification(metric_name="Average odds difference")
postprocess = CalibratedEqOdds()
postprocess.fit(new_dataset, predicted_dataset)
calib_dataset = postprocess.transform(predicted_dataset)


# exit()

# dataset = CompasDataset(r"D:\Users\peng\Desktop\Fairness_Projects\improve\compas-scores-two-years.csv", 
# target=['score_factor'], 
# features_to_drop=['c_charge_desc'], 
# features_to_keep=['two_year_recid', 'sex_female', 'score_factor',
#            'age_less_25', 'age_more_45',
#            'race_black', 'crime_M', 'priors_count'])
# X_train, X_test, Y_train, Y_test, Z_train, Z_test = dataset.split(0.75)
# # print(X_train)

# num_features = X_train.shape[1]
# train_size = X_train.shape[0]
# test_size = X_test.shape[0]
# # print(f"number of features: {num_features}")
# print(train_size, test_size)
# net = Net(num_features, 1, 0)

# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(net.parameters())
# epochs = 2000
# threshold = 0.5

# print("start training ....")
# yh = None
# max_acc = 0
# for i in range(epochs):
#     net.train()
#     x=torch.from_numpy(X_train).float()
#     y=torch.from_numpy(Y_train).float()
#     # y = y.squeeze(dim=1)
#     y_hat=net(x) 
#     loss=criterion(y_hat,y) # 计算损失
#     optimizer.zero_grad() # 前一步的损失清零
#     loss.backward() # 反向传播
#     optimizer.step() # 优化
#     if (i+1)%100 ==0 : # 这里我们每100次输出相关的信息
#         # 指定模型为计算模式
#         net.eval()
#         x = torch.from_numpy(X_test).float()
#         y = torch.from_numpy(Y_test).long()
#         # z = torch.from_numpy(Z_test).long()
#         loss = None
#         y_hat = None
#         with torch.no_grad():
#             y_hat=net(x)
#             # y = y.squeeze(dim=1)
#             loss = criterion(y_hat, y)
#             loss = loss.item()
#         pred = (y_hat > threshold).float()
#         # y=y.squeeze(dim=1)
#         correct = y.eq(pred).sum().item()
#         total = y.shape[0]
#         acc = correct / total *100
#         if acc > max_acc:
#             max_acc = acc
#             yh = y_hat
#         # z = z.squeeze(dim=1)
#         print(f"Epoch: {i}, Loss: {loss:.6f}, acc: {acc:.2f}%", )

# # exit()
# print(f"training finished with max accuracy: {max_acc}")
# yh = net(torch.Tensor(np.array(new_dataset.X)))
# predicted_dataset = copy.deepcopy(new_dataset)
# predicted_dataset.Y = yh.detach().cpu().numpy()

print("evaluating fairness of classifier after caliberation ....")
model_metrics = ModelMetrics(new_dataset, calib_dataset)
for metric in ModelMetrics.FAIRNESS_METRICS:
    m = model_metrics.group_fairness_metrics(metric)
    print(metric, ": ", m)