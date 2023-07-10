import os
import numpy as np
from typing import List, Optional
from typing import Union
import torch
from torch.nn import Module
from PIL import Image
import torchvision.transforms as transforms
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D, BatchNormalization, MaxPool2D, Input, Activation
from keras.regularizers import l2
# from keras.applications import VGG16
from tensorflow.keras.applications import VGG16
from art import config
from art.utils import load_mnist
from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd, add_single_bd
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, preprocess, load_cifar10
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from art.defences.transformer.poisoning.neural_cleanse import NeuralCleanse
from art.defences.transformer.transformer import Transformer

def conv2d_bn(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay)
                   )(x)
    layer = BatchNormalization()(layer)
    return layer


def conv2d_bn_relu(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = conv2d_bn(x, filters, kernel_size, weight_decay, strides)
    layer = Activation('relu')(layer)
    return layer


def ResidualBlock(x, filters, kernel_size, weight_decay, downsample=True):
    if downsample:
        # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
        residual_x = conv2d_bn(x, filters, kernel_size=1, strides=2)
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              strides=stride,
                              )
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         weight_decay=weight_decay,
                         strides=1,
                         )
    out = layers.add([residual_x, residual])
    out = Activation('relu')(out)
    return out


def ResNet18(classes, input_shape, weight_decay=1e-4):
    input = Input(shape=input_shape)
    x = input
    # x = conv2d_bn_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
    x = conv2d_bn_relu(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))

    # # conv 2
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 3
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 4
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 5
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input, x, name='ResNet18')
    return model

def save_jepg(adv_examples, output_path, adv_dataset):
    # 将torch加载的图像转换为PIL图像
    if adv_dataset == 'MNIST':
        image = adv_examples.squeeze()
    elif adv_dataset == 'CIFAR10':
        image = adv_examples
    pil_image = Image.fromarray(image)
    # 保存为JPEG图像
    pil_image.save(output_path, "JPEG")

class Transformerpoison(object):
    def __init__(self, model:Module, 
                mean:List[float]=[0.485, 0.456, 0.406], 
                std:List[float]=[0.229, 0.224, 0.225],
                adv_examples=None,
                adv_method: Optional[str] = 'PGD',
                adv_dataset: Optional[str] = 'CIFAR10',
                adv_nums: Optional[int] = 10,#
                device:Union[str, torch.device]='cuda',
                ):

        super(Transformerpoison, self).__init__()

        self.model = model
        self.norm = transforms.Normalize(mean, std)
        if adv_examples is None and (adv_method is None or adv_nums is None):
            raise Exception('Attack method and adversarial example nums need to be specified when the adversarial sample is not specified!')
        self.adv_examples = adv_examples
        self.adv_method = adv_method
        self.adv_dataset = adv_dataset
        self.adv_nums = adv_nums
        self.device = device
        # self.un_norm = UnNorm(mean, std)
        self.total_num = 0
        self.detect_num = 0
        self.detect_rate = 0
        self.no_defense_accuracy = 0

    def train(self, detect_poison:Transformer, norm):
        print('Read {} dataset (x_raw contains the original images):'.format(self.adv_dataset))
        if self.adv_dataset == 'CIFAR10':
            config.ART_DATA_PATH = './dataset/CIFAR10'
            load_dataset = load_cifar10
        elif self.adv_dataset == 'MNIST':
            config.ART_DATA_PATH = './dataset/MNIST'
            load_dataset = load_mnist
        (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_dataset(raw=True)

        print('Random Selection:')
        n_train = np.shape(x_raw)[0]
        num_selection = 10000 # 5000
        random_selection_indices = np.random.choice(n_train, num_selection)
        x_raw = x_raw[random_selection_indices]
        y_raw = y_raw[random_selection_indices]
        if self.adv_dataset == 'MNIST':
            BACKDOOR_TYPE = "pattern" # one of ['pattern', 'pixel', 'image']
        elif self.adv_dataset == 'CIFAR10':
            BACKDOOR_TYPE = "pattern" # one of ['pattern', 'pixel', 'image']
        max_val = np.max(x_raw)
        def add_modification(x):
                if BACKDOOR_TYPE == 'pattern':
                    return add_pattern_bd(x, pixel_value=max_val)
                elif BACKDOOR_TYPE == 'pixel':
                    return add_single_bd(x, pixel_value=max_val) 
                elif BACKDOOR_TYPE == 'image':
                    return insert_image(x, backdoor_path='../utils/dataset/backdoors/alert.png', size=(10,10))
                else:
                    raise("Unknown backdoor type")
        def poison_dataset(x_clean, y_clean, percent_poison, poison_func, dataset=None):
            x_poison = np.copy(x_clean)
            y_poison = np.copy(y_clean)
            is_poison = np.zeros(np.shape(y_poison))
            x_clean_save = x_clean[:10].copy()
            x_poison_save = np.copy(x_clean_save)
            if dataset != None:
                k = 0
            # sources=np.arange(10) # 0, 1, 2, 3, ...
            # targets=(np.arange(10) + 1) % 10 # 1, 2, 3, 4, ...
            sources = np.array([0])
            targets = np.array([1])
            for i, (src, tgt) in enumerate(zip(sources, targets)):
                n_points_in_tgt = np.size(np.where(y_clean == tgt))
                num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
                src_imgs = x_clean[y_clean == src]

                n_points_in_src = np.shape(src_imgs)[0]
                indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

                imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
                if dataset != None:
                    imgs_to_be_poisoned_save = np.copy(imgs_to_be_poisoned)
                backdoor_attack = PoisoningAttackBackdoor(poison_func)
                imgs_to_be_poisoned, poison_labels = backdoor_attack.poison(imgs_to_be_poisoned, y=np.ones(num_poison) * tgt)
                if dataset != None:
                    if k < 10:
                        for j in range(len(imgs_to_be_poisoned)):
                            x_clean_save[k] = imgs_to_be_poisoned_save[j]
                            x_poison_save[k] = imgs_to_be_poisoned[j]
                            k += 1
                            print(k)
                            if k == 10:
                                break
                x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
                y_poison = np.append(y_poison, poison_labels, axis=0)
                is_poison = np.append(is_poison, np.ones(num_poison))

            is_poison = is_poison != 0
            if dataset != None:
                image_num = min(np.sum(is_poison), 10)
                for i in range(image_num):
                    output_dir = "./output/backdoor/" + str(i)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    save_jepg(x_poison_save[i], output_dir + '/poisoned.jpeg', dataset)
                    save_jepg(x_clean_save[i], output_dir + '/clean.jpeg', dataset)
                    save_jepg(x_poison_save[i] - x_clean_save[i], output_dir + '/backdoor.jpeg', dataset)
                    if i == 0:
                        save_jepg(x_poison_save[i] - x_clean_save[i], './output/trigger/trigger.jpeg', dataset)

            return is_poison, x_poison, y_poison
        print('Poison training data:')
        percent_poison = .33

        n_train = np.shape(x_raw_test)[0]
        random_selection_indices = np.random.choice(n_train, 2000)
        x_raw_test = x_raw_test[random_selection_indices]
        y_raw_test = np.array(y_raw_test)[random_selection_indices]
        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = poison_dataset(x_raw, y_raw, percent_poison, add_modification)
        x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)
        print('Add channel axis:')
        if self.adv_dataset == 'MNIST':
            x_train = np.expand_dims(x_train, axis=3)

        print('Poison test data:')
        (is_poison_test, x_poisoned_raw_test, y_poisoned_raw_test) = poison_dataset(x_raw_test, y_raw_test, percent_poison, add_modification, dataset=self.adv_dataset)
        x_test, y_test = preprocess(x_poisoned_raw_test, y_poisoned_raw_test)
        print('Add channel axis:')
        if self.adv_dataset == 'MNIST':
            x_test = np.expand_dims(x_test, axis=3)

        print('Shuffle training data:')
        n_train = np.shape(y_train)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]
        print('Create Keras convolutional neural network - basic architecture from Keras examples:')
        if self.adv_dataset == 'CIFAR10':
            if self.model.__class__.__name__ == 'ResNet':
                model = ResNet18(input_shape=(32, 32, 3), classes=10, weight_decay=1e-4)
            elif self.model.__class__.__name__ == 'VGG':
                model = VGG16(input_shape=(32, 32, 3), weights = None, classes=10)
            else:
                raise Exception('CIFAR10 can only use ResNet18 and VGG16!')
        elif self.adv_dataset == 'MNIST':
            if self.model.__class__.__name__ == 'SmallCNN':
                model = Sequential()
                model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(10, activation='softmax'))
            elif self.model.__class__.__name__ == 'VGG':
                model = VGG16(input_shape=(28, 28, 1), weights = None, classes=10)
            else:
                raise Exception('MNIST can only use SmallCNN and VGG16!')

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        classifier = KerasClassifier(model=model, clip_values=(min_, max_))
        classifier.fit(x_train, y_train, nb_epochs=3, batch_size=128)
        clean_x_test = x_test[is_poison_test == 0]
        clean_y_test = y_test[is_poison_test == 0]
        poison_x_test = x_test[is_poison_test]
        poison_y_test = y_test[is_poison_test]

        with torch.no_grad():
            adv_predictions = classifier.predict(poison_x_test)
        no_defense_accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(poison_y_test, axis=1)) / len(poison_y_test)
        self.no_defense_accuracy = no_defense_accuracy
        cleanse = detect_poison(classifier)
        defence_cleanse = cleanse(classifier, steps=10, learning_rate=0.1, norm=norm)

        pattern, mask = defence_cleanse.generate_backdoor(poison_x_test, poison_y_test, np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
        trigger_predict = np.squeeze(mask * pattern)
        image = trigger_predict
        image_uint8 = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8)
        pil_image.save('./output/trigger/trigger_predict.jpeg', "JPEG")

        defence_cleanse.mitigate(clean_x_test, clean_y_test, mitigation_types=["filtering"])
        poison_pred = defence_cleanse.predict(poison_x_test)
        num_filtered = np.sum(np.all(poison_pred == np.zeros(10), axis=1))
        num_poison = len(poison_pred)
        effectiveness = float(num_filtered) / num_poison
        print(effectiveness)
        self.detect_rate = effectiveness
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method
        return attack_method, self.detect_num, self.detect_rate, self.no_defense_accuracy

    def print_res(self):
        print('detect rate: ', self.detect_rate)

class Neural_cleanse_l1(Transformerpoison):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        return self.train(NeuralCleanse, 1)
    
class Neural_cleanse_l2(Transformerpoison):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        return self.train(NeuralCleanse, 2)

class Neural_cleanse_linf(Transformerpoison):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        return self.train(NeuralCleanse, np.inf)