import numpy as np
import tensorflow as tf
import numpy as np


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#plt.switch_backend('agg')
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)


def build_resnet_dicts():
    resnet_dicts = {
        2: [0, 0, 0, 0],
        4: [1, 0, 0, 0],
        6: [1, 1, 0, 0],
        8: [1, 1, 1, 0],
        10: [1, 1, 1, 1],
        12: [1, 1, 2, 1],
        14: [1, 2, 2, 1],
        16: [1, 2, 2, 2],
        18: [2, 2, 2, 2],
        20: [2, 2, 3, 2],
        22: [2, 3, 3, 2],
        24: [2, 3, 4, 2],
        26: [2, 3, 4, 3],
        28: [3, 3, 4, 3],
        30: [3, 3, 5, 3],
        32: [3, 4, 5, 3]
    }
    for i in range(30):
        resnet_dicts[32+2 * i] = [3, 4, 5 + i, 3]
    return resnet_dicts

def build_mobilenet_dicts():
    mobilenet_dicts = {
        9: [1, 1, 1, 1],
        11: [1, 1, 2, 1],
        13: [1, 2, 2, 1],
        15: [2, 2, 2, 1],
        17: [2, 2, 2, 2],
        19: [2, 2, 3, 2],
    }
    for i in range(20):
        n = int(i/4)
        m = np.mod(i,4)
        mobilenet_dicts[21+2*i]=mobilenet_dicts[19+8*n].copy()
        for j in range(m+1):
            mobilenet_dicts[21+2*i][j]+=1
    return mobilenet_dicts


def build_vgg_dicts():
    vgg_dicts = {
        2: [1,1,1,1,1],
        3:[1,1,1,1,1],
        4:[1,1,1,1,1],
        5: [1,1,1,1,1],
        6: [2,1,1,1,1],
        7: [2,2,1,1,1],
        8: [2,2,2,1,1],
        9: [2,2,2,2,1],
        10: [2,2,2,2,2],
        11: [2,2,3,2,2],
        12: [2,2,3,3,2],
        13: [2,2,3,3,3],
    }
    for i in range(36):
        n = int(i/4)
        m = np.mod(i,4)
        vgg_dicts[14+i]=vgg_dicts[13+4*n].copy()
        for j in range(m+1):
            vgg_dicts[14+i][j]+=1
    return vgg_dicts


def gen_train(model, x_train, y_train, x_test, y_test, epochs, imgGen=None):
    name = model.name
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                                     patience=10, min_lr=1e-5)
    op = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=op,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['categorical_accuracy'])
    ear = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    if imgGen is None:
        history = model.fit(x_train, y_train, batch_size=128,
                            epochs=epochs, validation_data=(x_test, y_test), validation_freq=1
                            , callbacks=[reduce_lr],verbose=0
                            )
    else:
        history = model.fit(imgGen.flow(x_train, y_train, batch_size=128),
                            epochs=epochs, validation_data=(x_test, y_test), validation_freq=1
                            , callbacks=[reduce_lr],verbose=0
                            )
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    model.save("./output/cache/develop/model/" + name + ".h5")
    np.save('./output/cache/develop/data/' + name + '_loss', loss)
    np.save('./output/cache/develop/data/' + name + '_valloss', val_loss)
    np.save('./output/cache/develop/data/' + name + '_acc', acc)
    np.save('./output/cache/develop/data/' + name + '_valacc', val_acc)
    return model, acc, val_acc, loss, val_loss


def train_with_hypers(model, x_train, y_train, x_test, y_test, bs, lr, epochs, opname, imgGen=None):
    name = model.name + '_bs_' + str(bs) + '_lr_' + str(lr) + '_epo_' + str(epochs) + '_' + opname
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                                     patience=10, min_lr=1e-5)
    op = tf.keras.optimizers.Adam(learning_rate=lr) if opname == 'adam' else tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=op,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['categorical_accuracy'])
    if imgGen is None:
        history = model.fit(x_train, y_train, batch_size=bs,
                            epochs=epochs, validation_data=(x_test, y_test), validation_freq=1
                            , callbacks=[reduce_lr],verbose=0
                            )
    else:    
        history = model.fit(imgGen.flow(x_train, y_train, batch_size=bs),
                            epochs=epochs, validation_data=(x_test, y_test), validation_freq=1
                            , callbacks=[reduce_lr],verbose=0
                            )
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    model.save("./output/cache/develop/model/" + name + ".h5")
    np.save('./output/cache/develop/data/' + name + '_loss', loss)
    np.save('./output/cache/develop/data/' + name + '_valloss', val_loss) 
    np.save('./output/cache/develop/data/' + name + '_acc', acc)
    np.save('./output/cache/develop/data/' + name + '_valacc', val_acc)
    return model, acc, val_acc, loss, val_loss





def my_random_crop(image):
    image_new = np.zeros((40, 40, 3))
    sz = np.random.randint(9, size=2)
    image_new[4:36, 4:36, :] = image
    image = image_new[sz[0]:sz[0] + 32, sz[1]:sz[1] + 32, :]
    return image



def load_data():
    x_train = np.load('./output/cache/develop/tmp/xtr.npy')
    y_train = np.load('./output/cache/develop/tmp/ytr.npy')
    x_test = np.load('./output/cache/develop/tmp/xte.npy')
    y_test = np.load('./output/cache/develop/tmp/yte.npy')
    return x_train,y_train,x_test,y_test