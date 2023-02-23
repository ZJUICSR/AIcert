"""
Module providing adversarial attacks under a common interface.
"""
from art.attacks.attack import Attack, EvasionAttack, PoisoningAttack, PoisoningAttackBlackBox, PoisoningAttackWhiteBox
from art.attacks.attack import PoisoningAttackTransformer, ExtractionAttack, InferenceAttack, AttributeInferenceAttack
from art.attacks.attack import ReconstructionAttack

from art.attacks import evasion
from art.attacks import poisoning
