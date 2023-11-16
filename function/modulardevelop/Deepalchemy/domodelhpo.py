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
import pickle
import shutil
from tensorflow.keras import datasets
from tensorflow.keras.datasets import cifar10
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_model(deep, width):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    import numpy as np
    deep = {{uniform(dmin, dmax)}}
    if md != 'mobilenet':
        deep = np.round(deep / 2) * 2
    else:
        deep = np.round(deep / 2 + 0.5) * 2 - 1
    width = {{uniform(wmin, wmax)}}
    width = np.round(width * 4) / 4
    print('width='+str(width))
    print('depth='+str(deep))
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

    history = model.fit(x_train, y_train, batch_size,
                        epochs=epochs, validation_data=(x_test, y_test), validation_freq=1
                        , callbacks=[reduce_lr], verbose=2
                        )

    # del(imgGen)
    name = model.name + '_bs_' + str(batch_size) + '_lr_' + str(learning_rate) + '_epo_' + str(epochs) + '_' + opname
    model.save("./models/" + name + ".h5")
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    model.save("./models/" + name + ".h5")
    np.save('./data/' + name + '_loss', loss)
    np.save('./data/' + name + '_valloss', val_loss)
    np.save('./data/' + name + '_acc', acc)
    np.save('./data/' + name + '_valacc', val_acc)
    del (model)
    # del(x_train)
    # del(y_train)
    # del(x_test)
    # del(y_test)
    return {'loss': val_loss[-1], 'status': STATUS_OK, 'model': name}


def data():
    import tempparas
    x_train, y_train, x_test, y_test = tempparas.x_train, tempparas.y_train, tempparas.x_test, tempparas.y_test
    epochs = tempparas.epochs
    dmin = tempparas.dmin
    dmax = tempparas.dmax
    wmin = tempparas.wmin
    wmax = tempparas.wmax
    md = tempparas.md
    return x_train, y_train, x_test, y_test


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu',default='0')
    parser.add_argument('--times', default=1)
    parser.add_argument('--model', default='resnet')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    g = args.gpu
    t = args.times
    os.environ["CUDA_VISIBLE_DEVICES"] = str(g)
    if not os.path.exists('./data'):
        os.mkdir('./data')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    result, name, space = optim.minimize(model=create_model,
                                         data=data,
                                         algo=tpe.suggest,
                                         max_evals=int(t),
                                         trials=Trials(),
                                         verbose=False,
                                         return_space=True,
                                        keep_temp=False)
    valloss = np.load('./data/' + str(name) + '_valloss.npy')
    loss = np.load('./data/' + str(name) + '_loss.npy')
    valacc = np.load('./data/' + str(name) + '_valacc.npy')
    acc = np.load('./data/' + str(name) + '_acc.npy')
    np.save('./data/best.npy', valloss)
    history = {
        'loss': list(loss),
        'val_loss': list(valloss),
        'accuracy': list(acc),
        'val_accuracy': list(valacc)
    }
    output = open('../history_da.pkl', 'wb')
    pickle.dump(history, output)
    output.close()
    model = tf.keras.models.load_model('./models/' + str(name) + '.h5')
    model.save("./best.h5")
    

    print('The best model:')
    savedict = {}
    for i, hp in enumerate(result):
        if hp in ['deep', 'width']:
            savedict[hp] = np.round(result[hp])
        else:
            savedict[hp] = space[hp].pos_args[result[hp] + 1].obj

    output = open('./tmp/best_param.pkl', 'wb')
    pickle.dump(savedict, output)
    output.close()
    # os.remove('./tempparas.py')
    
