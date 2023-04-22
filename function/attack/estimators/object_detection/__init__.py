"""
Module containing estimators for object detection.
"""
from function.attack.estimators.object_detection.object_detector import ObjectDetectorMixin

from function.attack.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector
from function.attack.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
from function.attack.estimators.object_detection.pytorch_yolo import PyTorchYolo
from function.attack.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
