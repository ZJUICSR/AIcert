#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/06/30, ZJUICSR'

"""
This is the package's interface class for application.
"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(),"functions/attack/old"))
from adv.attack import Attack
import os.path as osp
from .torchattacks.attacks.fgsm import FGSM
from .torchattacks.attacks.bim import BIM
from .torchattacks.attacks.rfgsm import RFGSM
from .torchattacks.attacks.cw import CW
from .torchattacks.attacks.pgd import PGD
from .torchattacks.attacks.pgdl2 import PGDL2
from .torchattacks.attacks.eotpgd import EOTPGD
from .torchattacks.attacks.multiattack import MultiAttack
from .torchattacks.attacks.ffgsm import FFGSM
from .torchattacks.attacks.tpgd import TPGD
from .torchattacks.attacks.mifgsm import MIFGSM
from .torchattacks.attacks.vanila import VANILA
from .torchattacks.attacks.gn import GN
from .torchattacks.attacks.pgddlr import PGDDLR
from .torchattacks.attacks.apgd import APGD
from .torchattacks.attacks.apgdt import APGDT
from .torchattacks.attacks.fab import FAB
from .torchattacks.attacks.square import Square
from .torchattacks.attacks.autoattack import AutoAttack
from .torchattacks.attacks.onepixel import OnePixel
from .torchattacks.attacks.deepfool import DeepFool
from .torchattacks.attacks.sparsefool import SparseFool
from .torchattacks.attacks.difgsm import DI2FGSM
ROOT = osp.dirname(osp.abspath(__file__))

__all__ = [
    "FGSM",
    "BIM",
    "RFGSM",
    "CW",
    "PGD",
    "PGDL2",
    "EOTPGD",
    "MultiAttack",
    "FFGSM",
    "TPGD",
    "MIFGSM",
    "VANILA",
    "GN",
    "PGDDLR",
    "APGD",
    "APGDT",
    "FAB",
    "Square",
    "AutoAttack",
    "OnePixel",
    "DeepFool",
    "SparseFool",
    "DI2FGSM"
]



