
import torch
from torchvision import datasets, transforms
import numpy as np
import os
DEVICE = torch.device("cuda")
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def predict(img_array, model, data_name, device):
    # img_array = img_array/255.0
    if data_name == 'MNIST':
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
        x = img_array.copy()
        # x = x.reshape(28,28).astype(np.uint8)
        x = x.reshape(28,28)
        img_tensor = transform(x).unsqueeze(0)
        img_tensor = torch.tensor(img_tensor,dtype = torch.float32)
        img_tensor = img_tensor.to(DEVICE)
    elif data_name == 'CIFAR10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        x = img_array.copy()
        x = (x/255.0).reshape(32, 32, 3)
        # img_frame = Image.fromarray(x).convert("RGB")
        img_tensor = transform_test(x).unsqueeze(0)
        img_tensor = torch.tensor(img_tensor,dtype = torch.float32)
        img_tensor = img_tensor.to(DEVICE)
        # img_tensor = torch.unsqueeze(img_tensor, dim=0)

    elif data_name == 'ImageNet':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        x = img_array.copy()
        x = (x/255.0).reshape(224, 224, 3)
        # x = x.reshape(224, 224, 3).astype(np.uint8)
        # img_frame = Image.fromarray(x).convert("RGB")
        img_tensor = transform_test(x).unsqueeze(0)
        img_tensor = torch.tensor(img_tensor,dtype = torch.float32)
        img_tensor = img_tensor.to(DEVICE)
    #give the image and model,not their path!!
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)

    return predicted.item()

