import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pickle

with open('../result_m_v-1/normalized_data.pkl', 'rb') as f:# TODO: set the normalized dataset path here
    dataset = pickle.load(f)

tensor_x = torch.Tensor(dataset['x_test']) # transform to torch tensor
tensor_y = torch.Tensor(dataset['y_test'])

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader = DataLoader(my_dataset,batch_size=1) # create your dataloader

from tensorflow.keras.models import load_model
import autokeras as ak

model=torch.load('../result_m_v-1/best_model.pth').to(device) # TODO: set the torch model path here

lossf = nn.CrossEntropyLoss()
optim = optim.SGD(model.parameters(), lr=0.001)

model.eval()
loss = 0
total = 0
correct = 0
with torch.no_grad():
    for data, targets in my_dataloader:
        # data=data.reshape([data.shape[0],28,28])
        data = data.to(device)
        targets = targets.to(device)
        output =model(data)
        # loss += lossf(output, targets)
        correct += (output.argmax(1) == targets.argmax(1)).sum()
        #  for onnx2pytorch model, it has 3 outputs. 2nd output test acc=84.21%
        # for onnx2torch model, it has 95.58% acc, same with the original model.
        total += data.size(0)
        if total%3000==0:
            print(correct.item()/total)

acc = correct.item()/total
print(acc)
print(1)