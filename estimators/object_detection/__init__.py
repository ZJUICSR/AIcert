"""
Module containing estimators for object detection.
"""
from estimators.object_detection.object_detector import ObjectDetectorMixin

from estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector
from estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
from estimators.object_detection.pytorch_yolo import PyTorchYolo
from estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
