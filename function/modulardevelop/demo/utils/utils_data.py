'''
Author: your name
Date: 2021-08-06 08:47:23
LastEditTime: 2021-08-31 16:44:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /test_codes/utils/data.py
'''
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist,cifar10,fashion_mnist,cifar100
import tensorflow as tf


def mnist_load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def cifar10_load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def fashion_load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def cifar100_load_data():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 100)
    y_test = keras.utils.to_categorical(y_test, 100)
    return (x_train, y_train), (x_test, y_test)


def stl_load_data():
    # TODO: need to download stl data first.
    from scipy.io import loadmat
    from sklearn.preprocessing import LabelBinarizer
    train_raw = loadmat('/data1/zxy/DL_autokeras/1Autokeras/test_codes/utils/stl10_matlab/train.mat')
    test_raw = loadmat('/data1/zxy/DL_autokeras/1Autokeras/test_codes/utils/stl10_matlab/test.mat')
    train_images = np.array(train_raw['X']).reshape(-1, 3, 96, 96)
    test_images = np.array(test_raw['X']).reshape(-1, 3, 96, 96)
    train_images=np.transpose(train_images, (0, 3, 2, 1))
    test_images=np.transpose(test_images, (0, 3, 2, 1))
    train_labels = train_raw['y']
    test_labels = test_raw['y']
    
    # train_images = np.moveaxis(train_images, -1, 0)
    # test_images = np.moveaxis(test_images, -1, 0)
    
    print(train_images.shape)
    print(test_images.shape)
    train_images = train_images.astype('float64')
    test_images = test_images.astype('float64')
    train_labels = train_labels.astype('int64')
    test_labels = test_labels.astype('int64')
    train_images /= 255.0
    test_images /= 255.0
    # y_train = keras.utils.to_categorical(train_labels, 10)
    # y_test = keras.utils.to_categorical(test_labels, 10)
    lb = LabelBinarizer()
    y_train = lb.fit_transform(train_labels)
    y_test = lb.fit_transform(test_labels)

    return (train_images,y_train),(test_images,y_test)

def format_example(image, label,image_size=224):
    # image_size=256
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, (image_size, image_size))
    return image, label


if __name__=='__main__':
    # (x_train2, y_train2), (x_test2, y_test2)=emnist_load_data()
    # (x_train, y_train), (x_test, y_test)=svhn_load_data()
    # (x_train1, y_train1), (x_test1, y_test1)=cifar100_load_data()
    # (x_train, y_train), (x_test, y_test)=fashion_load_data()
    # print('finish test')

    import tensorflow_datasets as tfds
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # train = tfds.load('tiny_imagenet_dataset',data_dir='/home/zxy/workspace/DL_work/DL_autokeras/1Autokeras/test_codes/experiment/1_dataset',
    #             split='train',as_supervised=True)# shuffle_files=True, 

    # print(1)
    # # optional
    tf.compat.v1.enable_eager_execution()

    from tiny_imagenet import TinyImagenetDataset
    tiny_imagenet_builder = TinyImagenetDataset(data_dir='/home/zxy/workspace/DL_work/DL_autokeras/1Autokeras/test_codes/experiment/1_dataset')
    tiny_imagenet_builder.download_and_prepare()

    train_dataset = tiny_imagenet_builder.as_dataset(split="train")
    # validation_dataset = tiny_imagenet_builder.as_dataset(split="validation")
    print(1)

    assert(isinstance(train_dataset, tf.data.Dataset))
    # assert(isinstance(validation_dataset, tf.data.Dataset))

    for a_train_example in train_dataset.take(5):
        image, label, id = a_train_example["image"], a_train_example["label"], a_train_example["id"]
        print(f"Image Shape - {image.shape}")
        print(f"Label - {label.numpy()}")
        print(f"Id - {id.numpy()}")

    # print info about the data
    print(tiny_imagenet_builder.info)

    # ds = tfds.builder('food101',data_dir='/home/zxy/workspace/DL_work/DL_autokeras/1Autokeras/test_codes/experiment/1_dataset/')
    # ds = tfds.builder('cars196',data_dir='/home/zxy/workspace/DL_work/DL_autokeras/1Autokeras/test_codes/experiment/1_dataset/cars196')
    # ds.download_and_prepare()
    print(1)

    # # Load data from disk as tf.data.Datasets
    # datasets = mnist.as_dataset()
    # train_dataset, test_dataset = datasets['train'], datasets['test']
    # assert isinstance(train_dataset, tf.data.Dataset)

    # # And convert the Dataset to NumPy arrays if you'd like
    # for example in tfds.as_numpy(train_dataset):
    #   image, label = example['image'], example['label']
    #   assert isinstance(image, np.array)


    # (training_images, training_labels), (test_images, test_labels) =  \
    #     tfds.as_numpy(tfds.load('food101',data_dir='/data/zxy/DL_autokeras/1Autokeras/test_codes/experiment/1_dataset',
    #     split = ['train', 'test'],batch_size=-1, as_supervised=True))

    # ds = tfds.load('food101',data_dir='/data/zxy/DL_autokeras/1Autokeras/test_codes/experiment/1_dataset',
    #                split='train', shuffle_files=True)#,batch_size=32
    # train_batches = ds.shuffle(100).batch(32)
    # for example in tfds.as_numpy(ds):
    #     image, label = example['image'], example['label']
    #     assert isinstance(image, np.array)
