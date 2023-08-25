import torch

from typing import Union
from collections import OrderedDict

from ....src.utils import logger
from ....src.networks import SupportedWideResnetType


class ReshapeTeacherFCLayerMixin:
    """reshape teacher's fully connect layer in state_dict"""
    _device: Union[str, torch.device]
    model: SupportedWideResnetType

    def reshape_teacher_fc_layer(self, state_dict: OrderedDict):
        logger.debug(f"teacher fully connect weight: {state_dict['fc.weight'].shape}\n{state_dict['fc.weight']}")
        if state_dict.get("fc.bias") is not None:
            logger.debug(f"teacher fully connect bias: {state_dict['fc.bias'].shape}\n{state_dict['fc.bias']}")

        state_dict["fc.weight"] = torch.rand_like(self.model.fc.weight)
        logger.debug(f"reshaped fully connect weight: {state_dict['fc.weight'].shape}\n{state_dict['fc.weight']}")

        if state_dict.get("fc.bias") is not None:
            state_dict["fc.bias"] = torch.rand_like(self.model.fc.bias)
            logger.debug(f"reshaped fully connect bias: {state_dict['fc.bias'].shape}\n{state_dict['fc.bias']}")
