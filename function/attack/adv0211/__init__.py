import sys
import os
sys.path.append(os.path.join(os.getcwd(),"function/attack/adv0211"))
print(os.path.join(os.getcwd(),"function/attack/adv0211"))
from attack_api import EvasionAttacker, BackdoorAttacker

from art.mnist import Mnist