import sys
import os
sys.path.append(os.path.dirname(__file__))
import new_evaluation as eva
import tensorflow as tf
import numpy as np
#import myModel
import myModel
import time
import os
import argparse
import pickle
#from utils_data import *
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def get_model_n_params(model):
    return np.sum([tf.keras.backend.count_params(w) for w in (model.trainable_weights + model.non_trainable_weights)])


def get_resnet18_n_params(width, deep):
    resnet_dict = eva.build_resnet_dicts()
    model = myModel.resnet18(width, resnet_dict[deep])
    n = get_model_n_params(model)
    del model
    return n


def get_vgg_n_params(width, deep):
    vgg_dict = eva.build_vgg_dicts()
    model = myModel.VGG(width, vgg_dict[deep])
    n = get_model_n_params(model)
    del model
    return n


def get_mobilenet_n_params(width, deep):
    vgg_dict = eva.build_vgg_dicts()
    model = myModel.MobileNet(width, vgg_dict[deep])
    n = get_model_n_params(model)
    del model
    return n


def nparams_to_width(model, n, wmin, wmax, deep):
    if deep is None:
        deep = 18
    l = wmin
    r = wmax
    cnt_width = (l + r) / 2
    while np.abs(np.round(l * 2) / 2 - np.round(r * 2) / 2) > 0.5:
        if model == 'resnet':
            cnt_n = get_resnet18_n_params(cnt_width, deep)
        if model == 'vgg':
            cnt_n = get_vgg_n_params(cnt_width, deep)
        if model == 'mobilenet':
            cnt_n = get_mobilenet_n_params(cnt_width, deep)
        if cnt_n > n:
            r = cnt_width
        else:
            l = cnt_width
        cnt_width = (l + r) / 2

    return np.round(cnt_width * 2) / 2


def stop_OK(current_list, result_dict, nmax):
    cnt = 0
    n_list = [0, 0, 0]
    deep_list = [0, 0, 0]
    width_list = [0, 0, 0]
    for i, j in current_list:
        if result_dict.get((i, j)) is None:
            return False
        n_list[cnt] = get_resnet18_n_params(j, i)
        deep_list[cnt] = i
        width_list[cnt] = j
        cnt = cnt + 1
    if (max(deep_list) - min(deep_list) < 3) and (max(width_list) - min(width_list) < 3):
        return True
    else:
        return False


def bound_examination(model, nmax, bound_dict, width, deep):
    deep = round(deep)
    if model == 'resnet':
        if deep < 2:
            deep = 2
        elif deep >= 50:
            deep = 50
    elif model == 'vgg':
        if deep < 5:
            deep = 5
        elif deep >= 50:
            deep = 50
    elif model == 'mobilenet':
        if deep < 9:
            deep = 9
        elif deep >= 49:
            deep = 49
    if bound_dict.get(deep) is None:
        bound_dict[deep] = nparams_to_width(model, nmax, 0, 64, deep)
    if width < 1:
        width = 1
    elif width > bound_dict[deep]:
        width = bound_dict[deep]

    return bound_dict, width, deep


def NM_search_min(modeln,trainfunc, nmax, init_method, iternum):
    result_dict = {}
    history = []
    if init_method == 'rand':
        wrand1 = np.random.rand()
        wrand2 = np.random.rand()
        wrand3 = np.random.rand()
        drand1 = np.random.randint(1,25) *2
        drand2 = np.random.randint(1,25) *2
        drand3 = np.random.randint(1,25) *2
        # 初始三角形选定深度为18,18,34 宽度为nmax/3,2nmax/3,nmax/2对应的宽度
        current_list = [
            [drand1, nparams_to_width(nmax * wrand1, 0, 64, drand1)],
            [drand2, nparams_to_width(wrand2 * nmax, 0, 64, drand2)],
            [drand3, nparams_to_width(nmax * wrand3, 0, 64, drand3)],
        ]
        bound_dict = {
            drand1: nparams_to_width(nmax, 0, 64, drand1),
            drand2: nparams_to_width(nmax, 0, 64, drand2),
            drand3: nparams_to_width(nmax, 0, 64, drand3),
        }
    elif init_method == 'large':
        if modeln != 'mobilenet':
            bound_dict = {
                8: nparams_to_width(modeln, nmax, 0, 64, 8),
                42: nparams_to_width(modeln, nmax, 0, 64, 42)
            }

            current_list = [
                [8, nparams_to_width(modeln, nmax / 6, 0, 64, 8)],
                [8, nparams_to_width(modeln, 5 * nmax / 6, 0, 64, 8)],
                [42, nparams_to_width(modeln, nmax / 2, 0, 64, 42)],
            ]
            print('init: large')
        else:
            bound_dict = {
                13: nparams_to_width(modeln, nmax, 0, 64, 13),
                41: nparams_to_width(modeln, nmax, 0, 64, 41)
            }

            current_list = [
                [13, nparams_to_width(modeln, nmax / 6, 0, 64, 13)],
                [13, nparams_to_width(modeln, 5 * nmax / 6, 0, 64, 13)],
                [41, nparams_to_width(modeln, nmax / 2, 0, 64, 41)],
            ]
    elif init_method == 'normal':
        if modeln != 'mobilenet':
            bound_dict = {
                    18: nparams_to_width(modeln, nmax, 0, 64, 18),
                    34: nparams_to_width(modeln, nmax, 0, 64, 34)
                }
            current_list = [
                    [18, nparams_to_width(modeln, nmax / 3, 0, 64, 18)],
                    [18, nparams_to_width(modeln, 2 * nmax / 3, 0, 64, 18)],
                    [34, nparams_to_width(modeln, nmax / 2, 0, 64, 34)],
                ]
        else:
            bound_dict = {
                    21: nparams_to_width(modeln, nmax, 0, 64, 21),
                    35: nparams_to_width(modeln, nmax, 0, 64, 35)
                }

            current_list = [
                    [21, nparams_to_width(modeln, nmax / 3, 0, 64, 21)],
                    [21, nparams_to_width(modeln, 2 * nmax / 3, 0, 64, 21)],
                    [25, nparams_to_width(modeln, nmax / 2, 0, 64, 35)],
                ]
    iter = 0
    print('iter ' + str(iter))
    print('Current Singular:')
    for dp, wth in current_list:
        result_dict[(dp, wth)] = trainfunc(wth, dp)
        print('Deep = ' + str(dp) + ' Width = ' + str(wth) + ' Loss = ' + str(result_dict[(dp, wth)]))

    #while not stop_OK(current_list, result_dict, nmax):
    for times in range(iternum):
        iter = iter + 1
        print('iter ' + str(iter))
        current_list = sorted(current_list, key=lambda x: result_dict[tuple(x)])
        history.append([current_list[0],result_dict[tuple(current_list[0])]])
        if modeln == 'resnet':
            deep_o = round((current_list[0][0] + current_list[1][0]) / 4) * 2
        elif modeln == 'vgg':
            deep_o = round((current_list[0][0] + current_list[1][0]) / 2)
        else:
            deep_o = round((current_list[0][0] + current_list[1][0]) / 4 + 0.5) * 2 - 1
        width_o = round(current_list[0][1] + current_list[1][1]) / 2
        # result_dict[(deep_o, width_o)] = trainfunc(width_o, deep_o)

        deep_r = 2 * deep_o - current_list[2][0]
        if modeln == 'resnet':
            deep_r = round(deep_r / 2) * 2
        elif modeln == 'mobilenet':
            deep_r = round(deep_r / 2 + 0.5) * 2 - 1
        width_r = 2 * width_o - current_list[2][1]
        bound_dict, width_r, deep_r = bound_examination(modeln, nmax, bound_dict, width_r, deep_r)
        r = trainfunc(width_r, deep_r) if result_dict.get((deep_r, width_r)) is None else result_dict[(deep_r, width_r)]
        result_dict[(deep_r, width_r)] = r

        if result_dict[tuple(current_list[1])] > r >= result_dict[tuple(current_list[0])]:
            print('Reflection')
            current_list[2] = [deep_r, width_r]
        elif r < result_dict[tuple(current_list[0])]:
            print('Expansion')
            deep_e = deep_o + 2 * (deep_r - deep_o)
            width_e = width_o + 2 * (width_r - width_o)
            bound_dict, width_e, deep_e = bound_examination(modeln, nmax, bound_dict, width_e, deep_e)
            e = trainfunc(width_e, deep_e) if result_dict.get((deep_e, width_e)) is None else result_dict[(deep_e, width_e)]
            result_dict[(deep_e, width_e)] = e
            if e < r:
                current_list[2] = [deep_e, width_e]
            else:
                current_list[2] = [deep_r, width_r]
        elif r >= result_dict[tuple(current_list[1])]:
            deep_c = deep_o + 0.5 * (current_list[2][0] - deep_o)
            if modeln == 'resnet':
                deep_c = round(deep_c / 2) * 2
            elif modeln == 'mobilenet':
                deep_c = round(deep_c / 2 + 0.5) * 2 - 1
            width_c = width_o + 0.5 * (current_list[2][1] - width_o)
            bound_dict, width_c, deep_c = bound_examination(modeln, nmax, bound_dict, width_c, deep_c)
            c = trainfunc(width_c, deep_c) if result_dict.get((deep_c, width_c)) is None else result_dict[(deep_c, width_c)]
            result_dict[(deep_c, width_c)] = c
            if c < result_dict[tuple(current_list[2])]:
                print('Contraction')
                current_list[2] = [deep_c, width_c]
            else:
                print('Shrink')
                current_list[1][0] = current_list[0][0] + 0.5 * (current_list[1][0] - current_list[0][0])
                if modeln == 'resnet':
                    current_list[1][0] = round(current_list[1][0] / 2) * 2
                elif modeln == 'vgg':
                    current_list[1][0] = round(current_list[1][0])
                else:
                    current_list[1][0] = round(current_list[1][0] / 2 + 0.5) * 2 - 1
                current_list[1][1] = current_list[0][1] + 0.5 * (current_list[1][1] - current_list[0][1])
                current_list[2][0] = current_list[0][0] + 0.5 * (current_list[2][0] - current_list[0][0])
                if modeln == 'resnet':
                    current_list[2][0] = round(current_list[2][0] / 2) * 2
                elif modeln == 'vgg':
                    current_list[2][0] = round(current_list[2][0])
                else:
                    current_list[2][0] = round(current_list[2][0] / 2 + 0.5) * 2 - 1
                current_list[2][1] = current_list[0][1] + 0.5 * (current_list[2][1] - current_list[0][1])
                for dp, wth in current_list:
                    if result_dict.get((dp, wth)) is None:
                        result_dict[(dp, wth)] = trainfunc(wth, dp) 
        print('Current Singular:')
        for dp, wth in current_list:
            print('Deep = ' + str(dp) + ' Width = ' + str(wth) + ' Loss = ' + str(result_dict[(dp, wth)]))
    print(history)
    current_list = sorted(current_list, key=lambda x: result_dict[tuple(x)])
    dmin, dmax, wmin, wmax = min([k[0] for k in current_list]), max([k[0] for k in current_list]), min(
        [k[1] for k in current_list]), max([k[1] for k in current_list])
    if dmax == dmin:
        dmax += 1
    if wmax == wmin:
        wmax += 0.5
    return dmin, dmax, wmin, wmax
    #return current_list[0], result_dict[tuple(current_list[0])]


def calc_nmax(n_dataset, imggen_dict):
    n_max = n_dataset
    return n_max


def write_temp(key, data, model, epochs, **kwargs):
    # with open('./function/modulardevelop/tempparas.py','w') as f:
    with open('./output/cache/develop/tempparas.py','w') as f:
        f.write('import tensorflow as tf\n')
        # f.write('import datasets\n')
        f.write('import new_evaluation as eva\n')
        # f.write('from utils_data import *\n')
        f.write('md = \'' + model + '\'\n')

        if key == 0:
            f.write('dmin = '+str(kwargs['dmin'])+'\n')
            f.write('dmax = '+str(kwargs['dmax'])+'\n')
            f.write('wmin = '+str(kwargs['wmin'])+'\n')
            f.write('wmax = '+str(kwargs['wmax'])+'\n')
        else:
            f.write('d = '+str(kwargs['d'])+'\n')
            f.write('w = '+str(kwargs['w'])+'\n')    
        # if data == 'cifar10':
        #     f.write('(x_train, y_train), (x_test, y_test) = cifar10_load_data()\n')
        # elif data == 'mnist':
        #     f.write('(x_train, y_train), (x_test, y_test) = mnist_load_data()\n')
        # elif data == 'cifar100':
        #     f.write('(x_train, y_train), (x_test, y_test) = data_prepare_cifar100()\n')
        f.write('x_train, y_train, x_test, y_test = eva.load_data()\n')
        f.write('epochs = '+str(epochs)+'\n')

        f.close()


def gen_train_function(hpo,  gpu, modeln,epochs,data):
    # if data == 'cifar10':
    #     (x_train, y_train), (x_test, y_test) = cifar10_load_data()
    # elif data == 'mnist':
    #     (x_train, y_train), (x_test, y_test) = mnist_load_data()
    # elif data == 'cifar100':
    #     (x_train, y_train), (x_test, y_test) = data_prepare_cifar100()
    # else:
    [x_train, y_train, x_test, y_test] = data
    data = 'data'
    input_shape = x_train[0].shape
    nmax = y_train.shape[0]
    yn = y_train.shape[1]
    #use_imggen = 0
    if modeln == 'mobilenet':
        d0 = 21
    else:
        d0 = 18
    w0 = nparams_to_width(modeln,nmax, 0, 64, d0)
    nowdir = os.path.dirname(__file__)
    if not hpo:
        write_temp(1, data, modeln,epochs, d=d0, w=w0)
        # os.system('python ' + nowdir + '/center.py --gpu='+gpu+' --times=5 --model='+modeln)
        os.system('python ' + nowdir + '/center.py --gpu '+gpu+' --times 5 --model '+modeln)
        cdict = np.load(nowdir+'/center.npy',allow_pickle=True)[()]
        bs = cdict['batch_size']
        lr = cdict['learning_rate']
        opname = cdict['opname']
        # os.remove(nowdir+'/center.npy')
    def trainfunc(width, deep):
        if deep is None:
            deep = 18
        if modeln == 'resnet':
            model = myModel.resnet18(width, eva.build_resnet_dicts()[deep], out=yn, inp=input_shape)
        elif modeln == 'vgg':
            model = myModel.VGG(width, eva.build_vgg_dicts()[deep], out=yn, inp=input_shape)
        elif modeln == 'mobilenet':
            model = myModel.MobileNet(width, eva.build_mobilenet_dicts()[deep], out=yn, inp=input_shape)
        model, acc, vacc, loss, vloss = eva.train_with_hypers(model, x_train, y_train, x_test, y_test, bs, lr, epochs, opname)
        return vloss[-1]
    
    def trainfunc_hpo(dmin, dmax, wmin, wmax):

        write_temp(0, data, modeln, epochs, dmin=dmin, dmax=dmax, wmin=wmin, wmax=wmax)
        os.system('python ' + nowdir + '/domodelhpo.py --gpu='+gpu+' --times=5 --model='+modeln)
        valloss = np.load('./output/cache/develop/data/best.npy')
        return valloss
    return trainfunc if not hpo else trainfunc_hpo, nmax


def run(in_dict):

    gpu = in_dict['gpu']
    modelname = in_dict['modelname']
    dataset = in_dict['dataset']
    epochs = in_dict['epochs']
    init = in_dict['init']
    iternum = in_dict['iternum']
    time0 = time.time()
    trainfunc, nmax = gen_train_function(False, str(gpu), modelname, epochs,dataset)
    dmin, dmax, wmin, wmax = NM_search_min(modelname, trainfunc, nmax, init, iternum)
    trainfunc, nmax = gen_train_function(True, str(gpu), modelname,epochs,dataset)
    valloss = trainfunc(dmin, dmax, wmin, wmax)
    time1 = time.time()
    print('model vloss： ' + str(valloss))
    print('time use： ' + str(time1 - time0))


