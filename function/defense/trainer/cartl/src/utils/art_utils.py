from art.attacks.evasion import ProjectedGradientDescent, CarliniL2Method, CarliniLInfMethod
from art import attacks
from art.estimators.classification import PyTorchClassifier

from typing import Tuple, Dict, Optional

import numpy as np

from ...src import settings
from .data_utils import CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD, CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD, get_fc_out_features

import torch.nn as nn
import torch


def init_preprocessing(dataset: str):
    if dataset == "cifar100":
        mean = CIFAR100_TRAIN_MEAN
        std = CIFAR100_TRAIN_STD
    elif dataset == "cifar10":
        mean = CIFAR10_TRAIN_MEAN
        std = CIFAR10_TRAIN_STD
    else:
        raise ValueError(f"not supported dataset `{dataset}`")

    return (np.asarray(mean).reshape((3, 1, 1)),
            np.asarray(std).reshape((3, 1, 1)))


def init_classifier(
        *,
        model,
        preprocessing=None,
        input_shape: Tuple[int] = (3, 32, 32),
        nb_classes: Optional[int] = None,
        clip_values=(0, 1),
        train: bool = False
):
    if not nb_classes:
        nb_classes = get_fc_out_features(model)

    if not preprocessing:
        if nb_classes == 10:
            preprocessing = init_preprocessing("cifar10")
        elif nb_classes == 100:
            preprocessing = init_preprocessing("cifar100")
        else:
            raise ValueError("`nb_classes is not correct!`")

    classifier = PyTorchClassifier(
        model=model,
        preprocessing=preprocessing,
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=clip_values,
        loss=getattr(nn, settings.criterion)()
    )
    if not train:
        classifier.set_learning_phase(False)

    return classifier


def init_attacker(classifier: PyTorchClassifier, attacker_name: str, params: Dict):
    return getattr(attacks.evasion, attacker_name)(
        classifier,
        **params
    )


def evaluate_model_robustness(model: torch.nn.Module, x_test: np.ndarray, y_test: np.ndarray,
                              attacker_name: str, attacker_params: Dict, nb_classes: Optional[int] = None,
                              **kwargs) -> float:
    from time import time
    from pprint import pprint

    start = time()

    classifier = init_classifier(
        model=model,
        nb_classes=nb_classes,
        **kwargs
    )

    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
    print(f"Accuracy on benign test examples: {accuracy * 100:.3f}%")

    print(f"generate adversarial examples using attack: {attacker_name}")
    print("params for attack:")
    pprint(attacker_params)

    attacker = init_attacker(
        classifier=classifier,
        attacker_name=attacker_name,
        params=attacker_params
    )
    x_test_adv = attacker.generate(x=x_test)

    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
    print(f"Accuracy on adversarial test examples: {accuracy * 100:.2f}%")

    end = time()

    print(f"costing time: {end-start:.2f} secs")
    print("=" * 100)

    return accuracy


attack_params = {
    "ProjectedGradientDescent": {
        "train": {
            "eps": 8 / 255,
            "eps_step": 2 / 255,
            "batch_size": settings.batch_size,
            "max_iter": 7,
            "num_random_init": 1
        },
        "test": {
            "eps": 8 / 255,
            "eps_step": 2 / 255,
            "batch_size": settings.batch_size,
            "max_iter": 20,
            "num_random_init": 1
        }
    },
    "CarliniLInfMethod": {
        "batch_size": 128,
        "binary_search_steps": 10,
        "max_iter": 100,
        "eps": 0.03
    },
    "CarliniL2Method": {
        "batch_size": 128,
        "binary_search_steps": 10,
        "max_iter": 10,
    },
    "DeepFool": {
        "max_iter": 75,
        "batch_size": 128,
        "epsilon": 0.02
    },
    "FastGradientMethod": {
        # xz tql
        "eps": 8/255,
        "eps_step": 2/255,
        "batch_size": 128,
        "num_random_init": 1
    },
}
