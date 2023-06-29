import os
import torch
import numpy as np
import torchvision.transforms as transforms
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from typing import List, Optional
from typing import Union
from torch.nn import Module
from torchvision.models import vgg16
from PIL import Image

from sklearn.svm import SVC
from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd, add_single_bd
from art.estimators.classification import PyTorchClassifier
from art import config
from art.utils import load_mnist, preprocess, load_cifar10
from art.defences.detector.poison import ActivationDefence, ProvenanceDefense, SpectralSignatureDefense
from art.defences.detector.poison import PoisonFilteringDefence
from art.attacks.poisoning.poisoning_attack_svm import PoisoningAttackSVM
from art.estimators.classification.scikitlearn import SklearnClassifier, ScikitlearnSVC
import sys
sys.path.append('../../../..')
from function.defense.models import *

import logging
logging.getLogger('tensorflow').disabled = True

def save_jepg(adv_examples, output_path, adv_dataset):
    # 将torch加载的图像转换为PIL图像
    if adv_dataset == 'MNIST':
        image = adv_examples.squeeze()
    elif adv_dataset == 'CIFAR10':
        image = adv_examples.transpose(1, 2, 0)
    pil_image = Image.fromarray(image)
    # 保存为JPEG图像
    pil_image.save(output_path, "JPEG")

def generate_backdoor(
    x_clean, y_clean, percent_poison, backdoor_type="pattern", sources=np.arange(10), targets=(np.arange(10) + 1) % 10, dataset=None
):
    """
    Creates a backdoor in MNIST images by adding a pattern or pixel to the image and changing the label to a targeted
    class. Default parameters poison each digit so that it gets classified to the next digit.

    :param x_clean: Original raw data
    :type x_clean: `np.ndarray`
    :param y_clean: Original labels
    :type y_clean:`np.ndarray`
    :param percent_poison: After poisoning, the target class should contain this percentage of poison
    :type percent_poison: `float`
    :param backdoor_type: Backdoor type can be `pixel` or `pattern`.
    :type backdoor_type: `str`
    :param sources: Array that holds the source classes for each backdoor. Poison is
    generating by taking images from the source class, adding the backdoor trigger, and labeling as the target class.
    Poisonous images from sources[i] will be labeled as targets[i].
    :type sources: `np.ndarray`
    :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                    labeled as targets[i].
    :type targets: `np.ndarray`
    :return: Returns is_poison, which is a boolean array indicating which points are poisonous, x_poison, which
    contains all of the data both legitimate and poisoned, and y_poison, which contains all of the labels
    both legitimate and poisoned.
    :rtype: `tuple`
    """
    
    max_val = np.max(x_clean)
    y_clean = y_clean.ravel() #
    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    is_poison = np.zeros(np.shape(y_poison))
    x_clean_save = x_clean[:10].copy()
    x_poison_save = np.copy(x_clean_save)
    if dataset != None:
        k = 0
    for i, (src, tgt) in enumerate(zip(sources, targets)):
        n_points_in_tgt = np.size(np.where(y_clean == tgt))
        num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))

        src_imgs = x_clean[y_clean == src]

        n_points_in_src = np.shape(src_imgs)[0]
        indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

        imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
        if dataset != None:
            imgs_to_be_poisoned_save = np.copy(imgs_to_be_poisoned)
        if backdoor_type == "pattern":
            imgs_to_be_poisoned = add_pattern_bd(x=imgs_to_be_poisoned, pixel_value=max_val)
        elif backdoor_type == "pixel":
            imgs_to_be_poisoned = add_single_bd(imgs_to_be_poisoned, pixel_value=max_val)
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
        y_poison = np.append(y_poison, np.ones(num_poison) * tgt, axis=0)
        is_poison = np.append(is_poison, np.ones(num_poison))

    is_poison = is_poison != 0
    if dataset != None:
        image_num = min(np.sum(is_poison), 10)
        for i in range(image_num):
            output_dir = "/mnt/data2/yxl/AI-platform/output/backdoor/" + str(i)
            if not os.path.exists(output_dir):
                # 如果文件夹不存在，则创建文件夹
                os.makedirs(output_dir)
            save_jepg(x_poison_save[i], output_dir + '/poisoned.jpeg', dataset)
            save_jepg(x_clean_save[i], output_dir + '/clean.jpeg', dataset)
            save_jepg(x_poison_save[i] - x_clean_save[i], output_dir + '/backdoor.jpeg', dataset)

    return is_poison, x_poison, y_poison

class Detectpoison(object):
    def __init__(self, model:Module, 
                mean:List[float]=[0.485, 0.456, 0.406], 
                std:List[float]=[0.229, 0.224, 0.225],
                adv_examples=None,
                adv_method: Optional[str] = 'PGD',
                adv_dataset: Optional[str] = 'CIFAR10',
                adv_nums: Optional[int] = 10,#
                device:Union[str, torch.device]='cuda',
                ):

        super(Detectpoison, self).__init__()

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

    def train(self, detect_poison:PoisonFilteringDefence):
        print("Read {} dataset (x_raw contains the original images)".format(self.adv_dataset))
        if self.adv_dataset == 'CIFAR10':
            config.ART_DATA_PATH = '/mnt/data2/yxl/AI-platform/dataset/CIFAR10'
            load_dataset = load_cifar10
        elif self.adv_dataset == 'MNIST':
            config.ART_DATA_PATH = '/mnt/data2/yxl/AI-platform/dataset/MNIST'
            load_dataset = load_mnist
        (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_dataset(raw=True)
        # print(x_raw.shape, x_raw[0])
        n_train = np.shape(x_raw)[0]
        num_selection = 5000
        random_selection_indices = np.random.choice(n_train, num_selection)
        x_raw = x_raw[random_selection_indices]
        y_raw = y_raw[random_selection_indices]

        n_train = np.shape(x_raw_test)[0]
        num_selection = self.adv_nums
        random_selection_indices = np.random.choice(n_train, num_selection)
        x_raw_test = x_raw_test[random_selection_indices]
        y_raw_test = np.array(y_raw_test)[random_selection_indices]

        print("Poison training data")
        perc_poison = 0.33
        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = generate_backdoor(x_raw, y_raw, perc_poison)
        x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)

        if self.adv_dataset == 'MNIST':
            print("Add channel axis")
            x_train = np.expand_dims(x_train, axis=3)

        print("Poison test data")
        (is_poison_test, x_poisoned_raw_test, y_poisoned_raw_test) = generate_backdoor(x_raw_test, y_raw_test, perc_poison, dataset = self.adv_dataset)
        x_test, y_test = preprocess(x_poisoned_raw_test, y_poisoned_raw_test)
        print("Add channel axis")
        if self.adv_dataset == 'MNIST':
            x_test = np.expand_dims(x_test, axis=3)

        print("Shuffle training data so poison is not together")
        n_train = np.shape(y_train)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]
        is_poison_train = is_poison_train[shuffled_indices]

        if self.adv_dataset == 'CIFAR10':
            if self.model.__class__.__name__ == 'ResNet':
                model = ResNet18()
            elif self.model.__class__.__name__ == 'VGG':
                model = vgg16()
                model.classifier[6] = nn.Linear(4096, 10)
            else:
                raise Exception('CIFAR10 can only use ResNet18 and VGG16!')
            input_shape = (3, 32, 32)
        elif self.adv_dataset == 'MNIST':
            if self.model.__class__.__name__ == 'SmallCNN':
                model = SmallCNN()
            elif self.model.__class__.__name__ == 'VGG':
                model = vgg16()
                model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
                model.classifier[6] = nn.Linear(4096, 10)
            else:
                raise Exception('MNIST can only use SmallCNN and VGG16!')
            input_shape = (1, 28, 28)

        classifier = PyTorchClassifier(
            model=model,
            loss=torch.nn.CrossEntropyLoss(), 
            optimizer=torch.optim.Adam(model.parameters(), lr=0.1), 
            input_shape=input_shape, 
            nb_classes=10, 
            clip_values=(min_, max_)
        )
        x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
        x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
        classifier.fit(x_train, y_train, nb_epochs=4, batch_size=128) #nb_epochs=30

        with torch.no_grad():
            adv_predictions = classifier.predict(x_test)
        no_defense_accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        self.no_defense_accuracy = no_defense_accuracy

        print("Calling poisoning defence")
        defence = detect_poison(classifier, x_test, y_test)

        print("End-to-end method")
        defence.detect_poison(nb_clusters=3, nb_dims=9, reduce="PCA") #nb_dims=10

        print("Evaluate method when ground truth is known")
        is_clean = is_poison_test == 0
        confusion_matrix = defence.evaluate_defence(is_clean)
        
        jsonObject = json.loads(confusion_matrix)
        numerator = 0
        denominator = 0
        for label in jsonObject:
            numerator += jsonObject[label]['TruePositive']['numerator']
            numerator += jsonObject[label]['TrueNegative']['numerator']
            denominator += jsonObject[label]['TruePositive']['denominator']
            denominator += jsonObject[label]['TrueNegative']['denominator']

        self.detect_rate = numerator / denominator
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method
        return attack_method, self.detect_num, self.detect_rate, self.no_defense_accuracy

    def print_res(self):
        print('detect rate: ', self.detect_rate)

class Activation_defence(Detectpoison):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        return self.train(ActivationDefence)

class Spectral_signature(Detectpoison):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        return self.train(SpectralSignatureDefense)

class Provenance_defense(Detectpoison):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        print('Load and transform {} data'.format(self.adv_dataset))
        NB_TRAIN = 40
        NB_POISON = 5
        NB_VALID = 40
        NB_TRUSTED = 10
        NB_DEVICES = 10
        kernel = "linear"
        if self.adv_dataset == 'CIFAR10':
            config.ART_DATA_PATH = '/mnt/data2/yxl/AI-platform/dataset/CIFAR10'
            load_dataset = load_cifar10()
        elif self.adv_dataset == 'MNIST':
            config.ART_DATA_PATH = '/mnt/data2/yxl/AI-platform/dataset/MNIST'
            load_dataset = load_mnist()
        (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
        zero_or_four = np.logical_or(y_train == 4, y_train == 0, y_train == 9)
        x_train = x_train[zero_or_four]
        y_train = y_train[zero_or_four]
        tr_labels = np.zeros((y_train.shape[0], 2))
        tr_labels[y_train == 0] = np.array([1, 0])
        tr_labels[y_train == 4] = np.array([0, 1])
        y_train = tr_labels

        zero_or_four = np.logical_or(y_test == 4, y_test == 0)
        x_test = x_test[zero_or_four]
        y_test = y_test[zero_or_four]
        te_labels = np.zeros((y_test.shape[0], 2))
        te_labels[y_test == 0] = np.array([1, 0])
        te_labels[y_test == 4] = np.array([0, 1])
        y_test = te_labels

        n_samples_train = x_train.shape[0]
        n_features_train = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
        n_samples_test = x_test.shape[0]
        n_features_test = x_test.shape[1] * x_test.shape[2] * x_test.shape[3]

        x_train = x_train.reshape(n_samples_train, n_features_train)
        x_test = x_test.reshape(n_samples_test, n_features_test)
        x_train = x_train[:NB_TRAIN]
        y_train = y_train[:NB_TRAIN]

        trusted_data = x_test[:NB_TRUSTED]
        trusted_labels = y_test[:NB_TRUSTED]
        x_test = x_test[NB_TRUSTED:]
        y_test = y_test[NB_TRUSTED:]
        valid_data = x_test[:NB_VALID]
        valid_labels = y_test[:NB_VALID]
        x_test = x_test[NB_VALID:]
        y_test = y_test[NB_VALID:]
        print('assign random provenance features to the original training points')
        clean_prov = np.random.randint(NB_DEVICES - 1, size=x_train.shape[0])
        p_train = np.eye(NB_DEVICES)[clean_prov]
        no_defense = ScikitlearnSVC(model=SVC(kernel=kernel, gamma="auto"), clip_values=(min_, max_))
        no_defense.fit(x=x_train, y=y_train)
        print('poison a predetermined number of points starting at training points')
        poison_points = np.random.randint(no_defense._model.support_vectors_.shape[0], size=NB_POISON)
        all_poison_init = np.copy(no_defense._model.support_vectors_[poison_points])
        poison_labels = np.array([1, 1]) - no_defense.predict(all_poison_init)

        svm_attack = PoisoningAttackSVM(
            classifier=no_defense,
            x_train=x_train,
            y_train=y_train,
            step=0.1,
            eps=1.0,
            x_val=valid_data,
            y_val=valid_labels,
            verbose=False,
        )

        poisoned_data, _ = svm_attack.poison(all_poison_init, y=poison_labels)

        print('Stack on poison to data and add provenance of bad actor')
        all_data = np.vstack([x_train, poisoned_data])
        all_labels = np.vstack([y_train, poison_labels])
        is_poison = [0 for _ in range(len(x_train))] + [1 for _ in range(len(poisoned_data))]
        is_poison = np.array(is_poison)
        poison_prov = np.zeros((NB_POISON, NB_DEVICES))
        poison_prov[:, NB_DEVICES - 1] = 1
        all_p = np.vstack([p_train, poison_prov])
        print('Train clean classifier and poisoned classifier')
        model = SVC(kernel=kernel, gamma="auto")
        classifier = SklearnClassifier(model=model, clip_values=(min_, max_))

        classifier.fit(all_data, all_labels)
        
        with torch.no_grad():
            adv_predictions = classifier.predict(x_test)
        no_defense_accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        self.no_defense_accuracy = no_defense_accuracy

        defence_no_trust = ProvenanceDefense(classifier, all_data, all_labels, all_p, eps=0.1)
        
        # End-to-end method:
        # print("------------------- Results using size metric -------------------")
        # print(defence.get_params())
        defence = defence_no_trust
        defence.detect_poison()

        # Evaluate method when ground truth is known:
        is_clean = is_poison == 0
        confusion_matrix = defence.evaluate_defence(is_clean)
        # print("Evaluation defence results for size-based metric: ")
        jsonObject = json.loads(confusion_matrix)
        numerator = 0
        denominator = 0
        for label in jsonObject:
            numerator += jsonObject[label]['TruePositive']['numerator']
            numerator += jsonObject[label]['TrueNegative']['numerator']
            denominator += jsonObject[label]['TruePositive']['denominator']
            denominator += jsonObject[label]['TrueNegative']['denominator']
        self.detect_rate = numerator / denominator
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method
        return attack_method, self.detect_num, self.detect_rate, self.no_defense_accuracy
