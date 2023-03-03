import numpy as np
from keras.datasets import cifar10, mnist
from keras import utils

def load_cifar():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    # labels to categorical
    num_classes = 10
    train_y = utils.to_categorical(train_y, num_classes)
    test_y = utils.to_categorical(test_y, num_classes)

    return train_x, train_y, test_x, test_y


# Load data
def load_mnist():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # labels to categorical
    num_classes = 10
    Y_train = utils.to_categorical(Y_train, num_classes)
    Y_test = utils.to_categorical(Y_test, num_classes)

    return X_train, Y_train, X_test, Y_test
