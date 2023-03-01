from PIL import Image
import numpy as np
import os

def load_image_uint8(data_name,data_path):
    if data_name == 'MNIST':
        img = Image.open(data_path)
        x = np.array(img)
        x = x.reshape(28,28).astype(np.uint8)
    elif data_name == 'CIFAR10':
        img = Image.open(data_path)
        x = np.array(img)
        x = x.reshape(32,32,3).astype(np.uint8)
    elif data_name == 'ImageNet':
        img = Image.open(data_path)
        x = np.array(img)
        x = x.reshape(224,224,-1).astype(np.uint8)
    return x

def load_image(data_name,data_path):
    if data_name == 'MNIST':
        img = Image.open(data_path)
        x = np.array(img)
        x = x.reshape(28,28).astype(np.float32)
    elif data_name == 'CIFAR10':
        img = Image.open(data_path)
        x = np.array(img)
        x = x.reshape(32,32,3).astype(np.float32)
    elif data_name == 'ImageNet':
        img = Image.open(data_path)
        x = np.array(img)
        x = x.reshape(224,224,-1).astype(np.float32)
    return x


def save_to_image(img_array, data_name, save_path, file_name):
    if data_name == 'MNIST':
        img_array = img_array.copy()
        img_array = img_array.reshape(28,28)
        img_obj = Image.fromarray(img_array)
        # print(img_obj.shape)
        img_obj = img_obj.convert('L')
        # print(img_obj.shape)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_obj.save(os.path.join(save_path, file_name))
    elif data_name == 'CIFAR10':
        img_array = img_array.copy()
        img_array = img_array.reshape(32, 32,3)
        img_obj = Image.fromarray(img_array.astype('uint8')).convert('RGB')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_obj.save(os.path.join(save_path, file_name))
    elif data_name == 'ImageNet':
        img_array = img_array.copy()
        img_array = img_array.reshape(224, 224,-1)
        img_obj = Image.fromarray(img_array.astype('uint8')).convert('RGB')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_obj.save(os.path.join(save_path, file_name))

