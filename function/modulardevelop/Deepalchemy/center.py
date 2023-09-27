import sys
import os
sys.path.append(os.path.dirname(__file__))
from myModel import resnet18, VGG, MobileNet
import tensorflow as tf
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform
from hyperas import optim
from new_evaluation import build_resnet_dicts, build_vgg_dicts, build_mobilenet_dicts
import numpy as np

from tensorflow.keras import datasets
from tensorflow.keras.datasets import cifar10
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_model(deep, width):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    import numpy as np
    if md == 'resnet':
        model = resnet18(width, build_resnet_dicts()[deep], out=y_train.shape[1], inp=x_train[0].shape)
    elif md == 'vgg':
        model = VGG(width, build_vgg_dicts()[deep], out=y_train.shape[1], inp=x_train[0].shape)
    elif md == 'mobilenet':
        model = MobileNet(width, build_mobilenet_dicts()[deep], out=y_train.shape[1], inp=x_train[0].shape)
    batch_size = {{choice([64, 128, 256, 512])}}
    
    learning_rate = {{choice([1e-2, 1e-3, 1e-4])}}
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                                                     patience=5, min_lr=1e-5)

    opname = {{choice(['adam', 'sgd'])}}
    if opname == 'adam':
        op = tf.keras.optimizers.Adam(learning_rate)
    else:
        op = tf.keras.optimizers.SGD(learning_rate)
    model.compile(optimizer=op,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['categorical_accuracy'])
    ear = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')


    history = model.fit(x_train, y_train, batch_size,#steps_per_epoch=50000//epochs,
                            epochs=epochs, validation_data=(x_test, y_test), validation_freq=1
                            , callbacks=[reduce_lr], verbose=2
                            )
        #del(imgGen)
    name = model.name + '_bs_' + str(batch_size) + '_lr_' + str(learning_rate) + '_epo_' + str(epochs) + '_' + opname
    val_loss = history.history['val_loss']
    model.save("./models/" + name + ".h5")
    del(model)
    #del(x_train)
    #del(y_train)
    #del(x_test)
    #del(y_test)
    np.save('./data/' + name + '_valloss', val_loss)
    return {'loss': val_loss[-1], 'status': STATUS_OK, 'model': name}


def data():
    import tempparas
    x_train, y_train, x_test, y_test = tempparas.x_train, tempparas.y_train, tempparas.x_test, tempparas.y_test
    #imgGen = tempparas.imgGen
    epochs = tempparas.epochs
    width = tempparas.w
    deep = tempparas.d
    md = tempparas.md
    return x_train, y_train, x_test, y_test, imgGen


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu',required=True)
    parser.add_argument('--times',required=True)
    parser.add_argument('--model', default='resnet')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_args()
    g = args.gpu
    t = args.times
    os.environ["CUDA_VISIBLE_DEVICES"] = str(g)
    save_path = os.path.dirname(__file__)
    # print('here!')
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists('./models'):
        os.mkdir('./models')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    result, name, space = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=int(t),
                                          trials=Trials(),
                                          verbose=False,
                                          return_space=True)

    print('The best model:')
    savedict = {}
    for i, hp in enumerate(result):
        print(str(hp) + ':' + str(space[hp].pos_args[result[hp] + 1].obj))
        savedict[hp] = space[hp].pos_args[result[hp] + 1].obj
    np.save(save_path+'/center.npy', savedict)
    