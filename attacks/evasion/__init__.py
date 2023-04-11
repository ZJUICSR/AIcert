"""
Module providing evasion attacks under a common interface.
"""
# from attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch
# from attacks.evasion.adversarial_patch.adversarial_patch_numpy import AdversarialPatchNumpy
# from attacks.evasion.adversarial_patch.adversarial_patch_tensorflow import AdversarialPatchTensorFlowV2
# from attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch
# from attacks.evasion.adversarial_texture.adversarial_texture_pytorch import AdversarialTexturePyTorch
# from attacks.evasion.adversarial_asr import CarliniWagnerASR
from attacks.evasion.auto_attack import AutoAttack
from attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
# from attacks.evasion.brendel_bethge import BrendelBethgeAttack
from attacks.evasion.boundary import BoundaryAttack
from attacks.evasion.carlini import CarliniWagner
# from attacks.evasion.decision_tree_attack import DecisionTreeAttack
from attacks.evasion.deepfool import DeepFool
# from attacks.evasion.dpatch import DPatch
# from attacks.evasion.dpatch_robust import RobustDPatch
# from attacks.evasion.elastic_net import ElasticNet
from attacks.evasion.fast_gradient import FastGradientMethod
# from attacks.evasion.frame_saliency import FrameSaliencyAttack
# from attacks.evasion.feature_adversaries.feature_adversaries_numpy import FeatureAdversariesNumpy
# from attacks.evasion.feature_adversaries.feature_adversaries_pytorch import FeatureAdversariesPyTorch
# from attacks.evasion.feature_adversaries.feature_adversaries_tensorflow import FeatureAdversariesTensorFlowV2
from attacks.evasion.geometric_decision_based_attack import GeoDA
# from attacks.evasion.hclu import HighConfidenceLowUncertainty
from attacks.evasion.hop_skip_jump import HopSkipJump
# from attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleASR
# from attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPyTorch
from attacks.evasion.iterative_method import BasicIterativeMethod
# from attacks.evasion.laser_attack.laser_attack import LaserAttack
# from attacks.evasion.lowprofool import LowProFool
# from attacks.evasion.momentum_iterative_method import MomentumIterativeMethod
# from attacks.evasion.newtonfool import NewtonFool
# from attacks.evasion.pe_malware_attack import MalwareGDTensorFlow
from attacks.evasion.pixel_threshold import PixelAttack
from attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
# from attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
#     ProjectedGradientDescentNumpy,
# )
# from attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
#     ProjectedGradientDescentPyTorch,
# )
# from attacks.evasion.projected_gradient_descent.projected_gradient_descent_tensorflow_v2 import (
#     ProjectedGradientDescentTensorFlowV2,
# )
from attacks.evasion.over_the_air_flickering.over_the_air_flickering_pytorch import OverTheAirFlickeringPyTorch
from attacks.evasion.saliency_map import SaliencyMapMethod
# from attacks.evasion.shadow_attack import ShadowAttack
# from attacks.evasion.shapeshifter import ShapeShifter
from attacks.evasion.simba import SimBA
# from attacks.evasion.spatial_transformation import SpatialTransformation
from attacks.evasion.square_attack import SquareAttack
# from attacks.evasion.pixel_threshold import ThresholdAttack
from attacks.evasion.universal_perturbation import UniversalPerturbation
# from attacks.evasion.targeted_universal_perturbation import TargetedUniversalPerturbation
# from attacks.evasion.virtual_adversarial import VirtualAdversarialMethod
# from attacks.evasion.wasserstein import Wasserstein
from attacks.evasion.zoo import ZooAttack
# from attacks.evasion.sign_opt import SignOPTAttack
from attacks.evasion.gd_uap import GDUAP
from attacks.evasion.fastdrop import Fastdrop