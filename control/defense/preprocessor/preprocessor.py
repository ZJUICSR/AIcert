import numpy as np
from typing import List, Optional
from typing import Union
import tensorflow as tf
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist, load_cifar10
from art.defences.preprocessor import *
from art.defences.trainer import * 
from art.estimators.classification import KerasClassifier, PyTorchClassifier
from art.estimators.encoding.tensorflow import TensorFlowEncoder
from art.estimators.generation.tensorflow import TensorFlowGenerator
from art.defences.preprocessor.inverse_gan import InverseGAN
from control.defense.preprocessor.create_inverse_gan_models import build_gan_graph, build_inverse_gan_graph, load_model
from control.defense.models import *
from control.defense.utils.generate_aes import generate_adv_examples

class Prepro(object):
    def __init__(self, model:Module, 
                mean:List[float]=[0.485, 0.456, 0.406], 
                std:List[float]=[0.229, 0.224, 0.225],
                adv_examples=None,
                adv_method: Optional[str] = 'PGD',
                adv_dataset: Optional[str] = 'CIFAR10',
                adv_nums: Optional[int] = 10,#
                device:Union[str, torch.device]='cuda',
                ):

        super(Prepro, self).__init__()

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

    def generate_adv_examples(self):
        return generate_adv_examples(self.model, self.adv_method, self.adv_dataset, self.adv_nums, self.device)

    def load_adv_examples(self):
        data = torch.load(self.adv_examples)
        adv_dst = TensorDataset(data["x"].float().cpu(), data["y"].long().cpu())
        adv_loader = DataLoader(
        adv_dst,
        batch_size=1,
        shuffle=False,
        num_workers=2
        )
        return adv_loader

    def detect_base(self, preprocess_method:Preprocessor):
        if self.adv_examples is None:
            adv_imgs, cln_imgs, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, adv_labels = self.load_adv_examples() 
        preprocess = preprocess_method(clip_values=(0,1))
        adv_imgs_ss, _ = preprocess(adv_imgs.cpu().numpy()) #
        with torch.no_grad():
            predictions = self.model(adv_imgs)
            predictions_ss = self.model(torch.from_numpy(adv_imgs_ss).to(self.device))
        detect_rate = torch.sum(torch.argmax(predictions, dim = 1) != torch.argmax(predictions_ss, dim = 1)) / float(len(adv_imgs))
        self.detect_rate = float(detect_rate.cpu())
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate

    def train(self, preprocess_method:Preprocessor):
        print("Step 1: Load the {} dataset".format(self.adv_dataset))
        if self.adv_dataset == 'CIFAR10':
            (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
        elif self.adv_dataset == 'MNIST':
            (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
        x_test = x_test[:self.adv_nums]
        y_test = y_test[:self.adv_nums]
        ### 
        preprocess_raw = preprocess_method 
        preprocess = Preprocessor
        if preprocess_raw == LabelSmoothing:
            preprocess = preprocess_raw()
        else:
            preprocess = preprocess_raw(clip_values=(min_pixel_value, max_pixel_value))
        x_train, y_train = preprocess(x_train, y_train)
        print("Step 1a: Swap axes to PyTorch's NCHW format")
        x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
        x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

        print("Step 2: Create the model")
        if self.adv_dataset == 'CIFAR10':
            model = ResNet18()
        elif self.adv_dataset == 'MNIST':
            model = SmallCNN()

        print("Step 2a: Define the loss function and the optimizer")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        print("Step 3: Create the ART classifier")
        if self.adv_dataset == 'CIFAR10':
            input_shape = (3, 32, 32)
        elif self.adv_dataset == 'MNIST':
            input_shape = (1, 28, 28)
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=10,
        )

        print("Step 4: Train the ART classifier")
        classifier.fit(x_train, y_train, batch_size=256, nb_epochs=1)

        print("Step 5: Evaluate the ART classifier on benign test examples")

        predictions = classifier.predict(x_test)

        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))

        print("Step 6: Generate adversarial test examples")
        if self.adv_method == 'FGSM':
            attack = FastGradientMethod(estimator=classifier, eps=0.3)
        elif self.adv_method == 'PGD':
            attack = ProjectedGradientDescentPyTorch(estimator=classifier, eps=0.3, max_iter=40, eps_step=0.01)
        x_test_adv = attack.generate(x=x_test)

        print("Step 7: Evaluate the ART classifier on adversarial test examples")
        predictions_adv = classifier.predict(x_test_adv)

        accuracy = np.sum(np.argmax(predictions_adv, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) #
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
        self.detect_rate = accuracy
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate

    def print_res(self):
        print('detect rate: ', self.detect_rate)

# class Feature_squeezing(Prepro):
#     def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
#         super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
#     def detect(self):
#         return self.detect_base(FeatureSqueezing)

# class Jpeg_compression(Prepro):
#     def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
#         super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
#     def detect(self):
#         return self.detect_base(JpegCompression)

class Label_smoothing(Prepro):
    def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        return self.train(LabelSmoothing)

class Spatial_smoothing(Prepro):
    def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        return self.detect_base(SpatialSmoothing)

class Gaussian_augmentation(Prepro):
    def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        return self.train(GaussianAugmentation)

class Total_var_min(Prepro):
    def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        return self.detect_base(TotalVarMin)

class ModelImageMNIST(nn.Module):
    def __init__(self):
        super(ModelImageMNIST, self).__init__()
        self.fc = nn.Linear(28 * 28 * 1, 28 * 28 * 1 * 256)
    def forward(self, x):
        x = x.view(-1, 28 * 28 * 1)
        logit_output = self.fc(x)
        logit_output = logit_output.view(-1, 28, 28, 1, 256)
        return logit_output

class ModelImageCIFAR10(nn.Module):
    def __init__(self):
        super(ModelImageCIFAR10, self).__init__()
        self.fc = nn.Linear(32 * 32 * 1, 32 * 32 * 1 * 256)
    def forward(self, x):
        x = x.view(-1, 32 * 32 * 1)
        logit_output = self.fc(x)
        logit_output = logit_output.view(-1, 32, 32, 1, 256)
        return logit_output

def bpda(model, adv_dataset, examples, labels):
    from advertorch.attacks import LinfPGDAttack
    if adv_dataset == 'CIFAR10':
        eps = 0.031
        nb_iter = 20
        eps_iter = 0.003
    elif adv_dataset == 'MNIST':
        eps = 0.3
        nb_iter = 40
        eps_iter = 0.01
    adversary = LinfPGDAttack(
        model, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=eps,
        nb_iter=nb_iter, eps_iter=eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)
    adv_examples = adversary.perturb(examples, labels)
    return adv_examples

class Pixel_defend(Prepro):
    def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        if self.adv_examples is None:
            adv_imgs, cln_imgs, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, _ = self.load_adv_examples() 
        if self.adv_dataset == 'CIFAR10':
            model = ModelImageCIFAR10()
        elif self.adv_dataset == 'MNIST':
            model = ModelImageMNIST()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        if self.adv_dataset == 'CIFAR10':
            input_shape = (3, 32, 32)
        elif self.adv_dataset == 'MNIST':
            input_shape = (1, 28, 28)
        pixel_cnn = PyTorchClassifier(
            model=model, loss=loss_fn, optimizer=optimizer, input_shape=input_shape, nb_classes=10, clip_values=(0, 1)
        )
        preprocess = PixelDefend(eps=32, pixel_cnn=pixel_cnn)
        if self.adv_method == 'BPDA':
            defend_examples, _ = preprocess(cln_imgs.cpu().numpy()) #pixdefend处理
            defend_examples = torch.from_numpy(defend_examples).to(self.device) #转成tensor
            bpda_examples = bpda(self.model, self.adv_dataset, defend_examples, true_labels) #bpda攻击
            with torch.no_grad():
                predictions_bpda = self.model(bpda_examples)
            bpda_robustness = torch.sum(torch.argmax(predictions_bpda, dim = 1) == true_labels) / float(len(adv_imgs))
            self.detect_rate = float(bpda_robustness.cpu())
            # print('bpda_robustness:', float(bpda_robustness.cpu()))
        else:
            adv_imgs_ss, _ = preprocess(adv_imgs.cpu().numpy()) #
            with torch.no_grad():
                predictions = self.model(cln_imgs)
                predictions_adv = self.model(adv_imgs)
                predictions_ss = self.model(torch.from_numpy(adv_imgs_ss).to(self.device))
            accuracy = torch.sum(torch.argmax(predictions, dim = 1) == true_labels) / float(len(adv_imgs)) #0.9360000491142273 
            robustness = torch.sum(torch.argmax(predictions_adv, dim = 1) == true_labels) / float(len(adv_imgs)) #0.017000000923871994
            print('accuracy:', float(accuracy.cpu()), 'robustness:', float(robustness.cpu()))
            detect_rate = torch.sum(torch.argmax(predictions_ss, dim = 1) == true_labels) / float(len(adv_imgs)) #0.029000001028180122
            self.detect_rate = float(detect_rate.cpu())
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate

class Defense_gan(Prepro):
    def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        # load mnist data
        from .loaddata import load_mnist, load_cifar
        from .mnistmodel import mnist_model
        x_train, y_train, x_test, y_test = load_mnist()
        x_test = x_test[0:1000]
        y_test = y_test[0:1000]

        # load mnist CNN model in Keras
        # Mnist model
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        y = tf.placeholder(tf.float32, shape=(None, 10))
        mnist_model, logits = mnist_model(input_ph=x, logits=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        mnist_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        mnist_model.load_weights("/mnt/data2/yxl/AI-platform/control/defense/preprocessor/mnist_model.h5")
        classifier = KerasClassifier(model=mnist_model)

        # generate adversarial images using FGSM
        attack = FastGradientMethod(classifier, eps=0.13)
        X_adv = attack.generate(x_test)
        X_adv = np.clip(X_adv, 0, 1)

        # accuracy
        # preds_x_test = np.argmax(classifier.predict(x_test), axis=1)
        # acc = np.sum(preds_x_test == np.argmax(y_test, axis=1)) / y_test.shape[0]
        # fooling rate
        probs_X_adv = classifier.predict(X_adv)
        preds_X_adv = np.argmax(probs_X_adv, axis=1)
        fooling_rate = np.sum(preds_X_adv != np.argmax(y_test, axis=1)) / y_test.shape[0]
        sess = tf.Session()
        gen_tf, enc_tf, z_ph, image_to_enc_ph = load_model(sess, "model-dcgan", "/mnt/data2/yxl/AI-platform/utils/resources/models/tensorflow1") # model tarained with 10 epochs
        gan = TensorFlowGenerator(input_ph=z_ph, model=gen_tf, sess=sess,)
        inverse_gan = TensorFlowEncoder(input_ph=image_to_enc_ph, model=enc_tf, sess=sess,)
        preproc = InverseGAN(sess=sess, gan=gan, inverse_gan=None)
        X_def, _ = preproc(X_adv, maxiter=20)
        preds_X_def = np.argmax(classifier.predict(X_def), axis=1)
        fooling_rate = np.sum(preds_X_def == np.argmax(y_test, axis=1)) / y_test.shape[0]
        # fooling_rate = np.sum(preds_X_def != preds_X_adv) / y_test.shape[0]
        # logger.info('Fooling rate after Defense GAN: %.2f%%', (fooling_rate  * 100))
        self.detect_rate = fooling_rate
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate

class Inverse_gan(Prepro):
    def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        # load mnist data
        from .loaddata import load_mnist, load_cifar
        from .mnistmodel import mnist_model
        x_train, y_train, x_test, y_test = load_mnist()
        x_test = x_test[0:1000]
        y_test = y_test[0:1000]

        # load mnist CNN model in Keras
        # Mnist model
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        y = tf.placeholder(tf.float32, shape=(None, 10))
        mnist_model, logits = mnist_model(input_ph=x, logits=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        mnist_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        mnist_model.load_weights("/mnt/data2/yxl/AI-platform/control/defense/preprocessor/mnist_model.h5")
        classifier = KerasClassifier(model=mnist_model)

        # generate adversarial images using FGSM
        attack = FastGradientMethod(classifier, eps=0.13)
        X_adv = attack.generate(x_test)
        X_adv = np.clip(X_adv, 0, 1)

        # accuracy
        # preds_x_test = np.argmax(classifier.predict(x_test), axis=1)
        # acc = np.sum(preds_x_test == np.argmax(y_test, axis=1)) / y_test.shape[0]
        # fooling rate
        probs_X_adv = classifier.predict(X_adv)
        preds_X_adv = np.argmax(probs_X_adv, axis=1)
        fooling_rate = np.sum(preds_X_adv != np.argmax(y_test, axis=1)) / y_test.shape[0]
        sess = tf.Session()
        gen_tf, enc_tf, z_ph, image_to_enc_ph = load_model(sess, "model-dcgan", "/mnt/data2/yxl/AI-platform/utils/resources/models/tensorflow1") # model tarained with 10 epochs
        gan = TensorFlowGenerator(input_ph=z_ph, model=gen_tf, sess=sess,)
        inverse_gan = TensorFlowEncoder(input_ph=image_to_enc_ph, model=enc_tf, sess=sess,)
        preproc = InverseGAN(sess=sess, gan=gan, inverse_gan=None)
        X_def, _ = preproc(X_adv, maxiter=20)
        preds_X_def = np.argmax(classifier.predict(X_def), axis=1)
        fooling_rate = np.sum(preds_X_def == np.argmax(y_test, axis=1)) / y_test.shape[0]
        # fooling_rate = np.sum(preds_X_def != preds_X_adv) / y_test.shape[0]
        # logger.info('Fooling rate after Inverse GAN: %.2f%%', (fooling_rate  * 100))
        self.detect_rate = fooling_rate
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate