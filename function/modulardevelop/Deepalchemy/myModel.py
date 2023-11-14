from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, Activation, Input, DepthwiseConv2D, GlobalAveragePooling2D
import tensorflow as tf
import numpy as np



def LeNet5():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=[28, 28, 1]),
        Conv2D(filters=6, kernel_size=(5, 5),
               activation='sigmoid'),
        MaxPool2D(pool_size=(2, 2), strides=2),

        Conv2D(filters=16, kernel_size=(5, 5),
               activation='sigmoid'),
        MaxPool2D(pool_size=(2, 2), strides=2),

        Flatten(),
        Dense(120, activation='sigmoid'),
        Dense(84, activation='sigmoid'),
        Dense(10, activation='softmax')])

    return model


def VGG16():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=[32, 32, 3]),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same', ),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        Dropout(0.2),

        Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        Dropout(0.2),

        Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        Dropout(0.2),

        Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        Dropout(0.2),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])

    return model


def VGG(w, block_list, out=None, inp=None):
    if out is None:
        out = 10
    if inp is None:
        inp = (32, 32, 3)
        
    inputt = Input(shape=inp)
    x = inputt
    for block_id in range(len(block_list) - 1):
        for layer_id in range(block_list[block_id]):
            x = Conv2D(filters=int(np.round(w)), kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Dropout(0.2)(x)
        w = w * 2

    x = Flatten()(x)
    for layer_id in range(block_list[-1] - 1):
        x = Dense(w, activation='relu')(x)
        x = Dropout(0.2)(x)
    outt = Dense(out, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputt, outputs=outt, name='vgg' + str(np.sum(block_list)) + '_cifar10_' + str(w))

    return model


def ResnetBlock(x, filters, strides=1, residual_path=False):
    res = x
    c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)(x)
    b1 = BatchNormalization()(c1)
    a1 = Activation('relu')(b1)

    c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)(a1)
    y = BatchNormalization()(c2)

    if residual_path:
        res = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)(res)
        res = BatchNormalization()(res)

    out = Activation('relu')(y + res)
    return out


def resnet18(out_filters, block_list=None, out=None, inp=None):
    if block_list is None:
        block_list = [2, 2, 2, 2]
    if out is None:
        out = 10
    if inp is None:
        inp = (32, 32, 3)
    input = Input(shape=inp)
    c1 = Conv2D(int(np.round(out_filters)), (3, 3), strides=1, padding='same', use_bias=False)(input)
    b1 = BatchNormalization()(c1)
    x = Activation('relu')(b1)


    for block_id in range(len(block_list)):  # 第几个resnet block
        for layer_id in range(block_list[block_id]):  # 第几个卷积层
            if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                x = ResnetBlock(x, int(np.round(out_filters)), strides=2, residual_path=True)
            else:
                x = ResnetBlock(x, int(np.round(out_filters)), residual_path=False)
        out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
    p1 = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = Dense(out, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())(p1)

    model = tf.keras.Model(inputs=input, outputs=out, name='resnet'+str(np.sum(block_list)*2+2)+'_'+str(out_filters))
    #tf.keras.utils.plot_model(model, 'resnet18.png')
    return model


def ConvBNRelu(x, filters, size, strides=1, padding='valid'):
    x = Conv2D(filters, size, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def InceptionA(x, pool_features):
    branch1 = ConvBNRelu(x, 64, (1, 1))

    branch5 = ConvBNRelu(x, 48, (1, 1))
    branch5 = ConvBNRelu(branch5, 64, (5, 5), 1, 'same')

    branch3 = ConvBNRelu(x, 64, (1, 1))
    branch3 = ConvBNRelu(branch3, 96, (3, 3), 1, 'same')
    branch3 = ConvBNRelu(branch3, 96, (3, 3), 1, 'same')

    branchpool = tf.keras.layers.AveragePooling2D((3, 3), 1, 'same')(x)
    branchpool = ConvBNRelu(branchpool, pool_features, (3, 3), 1, 'same')

    y = tf.keras.layers.concatenate([branch1, branch5, branch3, branchpool], axis=3)
    return y


def InceptionB(x):
    branch3 = ConvBNRelu(x, 384, (3, 3), 2)

    branch33 = ConvBNRelu(x, 64, (1, 1))
    branch33 = ConvBNRelu(branch33, 96, (3, 3), 1, 'same')
    branch33 = ConvBNRelu(branch33, 96, (3, 3), 2)

    branchpool = MaxPool2D((3, 3), 2)(x)
    y = tf.keras.layers.concatenate([branch3, branch33, branchpool], axis=3)

    return y


def InceptionC(x, c7s):
    branch1 = ConvBNRelu(x, 192, (1, 1))

    branch7 = ConvBNRelu(x, c7s, (1, 1))
    branch7 = ConvBNRelu(branch7, c7s, (7, 1), 1, 'same')
    branch7 = ConvBNRelu(branch7, 192, (1, 7), 1, 'same')

    branch77 = ConvBNRelu(x, c7s, (1, 1))
    branch77 = ConvBNRelu(branch77, c7s, (7, 1), 1, 'same')
    branch77 = ConvBNRelu(branch77, c7s, (1, 7), 1, 'same')
    branch77 = ConvBNRelu(branch77, c7s, (7, 1), 1, 'same')
    branch77 = ConvBNRelu(branch77, 192, (1, 7), 1, 'same')

    branchpool = tf.keras.layers.AveragePooling2D((3, 3), 1, 'same')(x)
    branchpool = ConvBNRelu(branchpool, 192, (1, 1))

    y = tf.keras.layers.concatenate([branch1, branch7, branch77, branchpool], axis=3)
    return y


def InceptionD(x):
    branch3 = ConvBNRelu(x, 192, (1, 1))
    branch3 = ConvBNRelu(branch3, 320, (3, 3), 2)

    branch7 = ConvBNRelu(x, 192, (1, 1))
    branch7 = ConvBNRelu(branch7, 192, (7, 1), 1, 'same')
    branch7 = ConvBNRelu(branch7, 192, (1, 7), 1, 'same')
    branch7 = ConvBNRelu(branch7, 192, (3, 3), 2)

    branchpool = tf.keras.layers.AveragePooling2D((3, 3), 2)(x)

    y = tf.keras.layers.concatenate([branch3, branch7, branchpool], axis=3)
    return y


def InceptionE(x):
    branch1 = ConvBNRelu(x, 320, (1, 1))

    branch3 = ConvBNRelu(x, 384, (1, 1))
    branch3a = ConvBNRelu(branch3, 384, (1, 3), 1, 'same')
    branch3b = ConvBNRelu(branch3, 384, (3, 1), 1, 'same')
    branch3 = tf.concat(axis=3, values=[branch3a, branch3b])

    branch33 = ConvBNRelu(x, 448, (1, 1))
    branch33 = ConvBNRelu(branch33, 384, (3, 3), 1, 'same')
    branch33a = ConvBNRelu(branch33, 384, (1, 3), 1, 'same')
    branch33b = ConvBNRelu(branch33, 384, (3, 1), 1, 'same')
    branch33 = tf.concat(axis=3, values=[branch33a, branch33b])

    branchpool = tf.keras.layers.AveragePooling2D((3, 3), 1, 'same')(x)
    branchpool = ConvBNRelu(branchpool, 192, (1, 1))

    y = tf.keras.layers.concatenate([branch1, branch3, branch33, branchpool], axis=3)

    return y


def Inceptionv3():
    input = Input(shape=(32, 32, 3))
    c1 = ConvBNRelu(input, 32, (3, 3), 1, 'same')
    c2 = ConvBNRelu(c1, 32, (3, 3), 1, 'same')
    c3 = ConvBNRelu(c2, 64, (3, 3), 1, 'same')
    c4 = ConvBNRelu(c3, 80, (1, 1))
    c5 = ConvBNRelu(c4, 80, (3, 3))

    A1 = InceptionA(c5, 32)
    A2 = InceptionA(A1, 64)
    A3 = InceptionA(A2, 64)

    B1 = InceptionB(A3)

    C1 = InceptionC(B1, 128)
    C2 = InceptionC(C1, 160)
    C3 = InceptionC(C2, 160)
    C4 = InceptionC(C3, 192)

    D1 = InceptionD(C4)

    E1 = InceptionE(D1)
    E2 = InceptionE(E1)

    avgpool = tf.keras.layers.GlobalAveragePooling2D()(E2)
    y = Dense(10, activation='softmax')(avgpool)

    model = tf.keras.Model(inputs=input, outputs=y)
    # tf.keras.utils.plot_model(model, 'resnet18.png')
    return model


def depthwise_separable(x, params):
    # f1/f2 filter size, s1 stride of conv
    (s1, f2) = params
    x = DepthwiseConv2D((3, 3), strides=(s1[0], s1[0]), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(f2[0]), (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def MobileNet(w, block_list, out=None, inp=None):
    if out is None:
        out = 10
    if inp is None:
        inp = (32, 32, 3)
    img_input = Input(shape=inp)
    x = Conv2D(int(32), (3, 3), strides=(2, 2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for block_id in range(len(block_list) - 1):
        for layer_id in range(block_list[block_id] - 1):
            x = depthwise_separable(x, params=[(1,), (w,)])
        x = depthwise_separable(x, params=[(2,), (2 * w,)])
        w = w * 2
    x = depthwise_separable(x, params=[(2,), (w,)])
    for layer_id in range(block_list[-1] - 1):
        x = depthwise_separable(x, params=[(1,), (w,)])

    x = GlobalAveragePooling2D()(x)
    output = Dense(out, activation='softmax')(x)

    model = tf.keras.Model(img_input, output, name='mbnet' + str(np.sum(block_list)) + '_cifar10_' + str(w))
    return model