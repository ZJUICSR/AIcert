from typing import Union, List

import torch

from .resnet import ResNet
from .parseval_resnet import ParsevalResNet
from .wrn import WideResNet
from .parseval_wrn import ParsevalWideResNet
from ...src.networks import SupportedWideResnetType, SupportedResnetType, SupportedAllModuleType


class WRN34Block:
    """divide wrn34 model into 17 blocks,
    details can be found in paper `ADVERSARIALLY ROBUST TRANSFER LEARNING`"""

    def __init__(self, model: SupportedWideResnetType):
        self.model = model
        self._set_block()

        self._total_blocks = 17

    def get_block(self, num: int):
        if 1 <= num <= 15:
            block_num, layer_num = divmod(num+4, 5)
            return getattr(self.model, f"block{block_num}").layer[layer_num]
        elif num == 16:
            return torch.nn.Sequential(self.model.bn1, self.model.relu)
        elif num == 17:
            return torch.nn.Sequential(self.model.fc)
        else:
            raise ValueError(f"unexpected block number: {num}")

    def get_total_blocks(self) -> int:
        return self._total_blocks

    def _set_block(self):
        for i in range(1, 18):
            setattr(self, f"block{i}", self.get_block(i))

class WRNBlocks:
    """
        Divide WRN into blocks. (WRN D-W: D = 4 + n * 6)
             [conv 16]
                |
        [conv160, conv160] * n
                |
        [conv320, conv320] * n
                |
        [conv640, conv640] * n
                |
           [bn, avg_pool]
                |
              [fc]
    """
    def __init__(self, model: SupportedWideResnetType):
        self._model = model
        assert isinstance(model.block1.layer, torch.nn.Sequential)
        self._n = len(model.block1.layer)
        self._total_blocks = 3 * self._n + 2
        self._set_block()
    
    def get_block(self, num: int):
        # conv blocks
        if 1 <= num <= (3 * self._n):
            block_num, layer_num = divmod(num + self._n - 1, self._n)
            return getattr(self._model, f"block{block_num}").layer[layer_num]
        # the penultimate is 'bn' and 'relu' layer
        elif num == self._total_blocks - 1:
            return torch.nn.Sequential(self._model.bn1, self._model.relu)
        # last block is 'fc' layer
        elif num == self._total_blocks:
            return torch.nn.Sequential(self._model.fc)
        else:
            raise ValueError(f"unexpected block number: {num}")
    
    def get_total_blocks(self) -> int:
        return self._total_blocks

    def _set_block(self):
        for i in range(1, self._total_blocks + 1):
            setattr(self, f"block{i}", self.get_block(i))


class ResnetBlocks:

    def __init__(self, model: SupportedAllModuleType) -> None:
        self.model = model
        self._block_num_list = self.get_block_num_list()
        # 1 indicates fc layer
        self._total_blocks = sum(self._block_num_list) + 1

        self._set_block_attr()

    def get_block_num_list(self) -> List[int]:
        block_num_list = []
        for i in range(2, 6):
            blocks = getattr(self.model, f"conv{i}_x")
            block_num_list.append(len(blocks))

        return block_num_list

    def get_total_blocks(self) -> int:
        return self._total_blocks

    def get_block(self, num: int):
        if 1 <= num < self._total_blocks:
            for i in range(len(self._block_num_list)):
                if num <= self._block_num_list[i]:
                    return getattr(self.model, f"conv{i+2}_x")[num-1]
                num -= self._block_num_list[i]
        elif num == self.get_total_blocks():
            return torch.nn.Sequential(self.model.fc)
        else:
            raise ValueError(f"unexpected block number: {num}")

    def _set_block_attr(self):
        for i in range(1, self._total_blocks+1):
            setattr(self, f"block{i}", self.get_block(i))


class Resnet18Block:
    """divided resnet into 9 blocks

    Blocks:
        block1-2: residual block of conv2_x
        block3-4: residual block of conv3_x
        block5-6: residual block of conv4_x
        block7-8: residual block of conv5_x
        block9: fully connect layer
    """

    def __init__(self, model: SupportedResnetType):
        self.model = model
        self._set_blocks()

        self._total_blocks = 9

    def get_block(self, num: int):
        if 1 <= num <= 8:
            conv_num, residual_num = divmod(num+3, 2)
            conv_block = getattr(self.model, f"conv{conv_num}_x")
            # return torch.nn.Sequential(conv_block[residual_num])
            return conv_block[residual_num]
        elif num == 9:
            return torch.nn.Sequential(self.model.fc)
        else:
            raise ValueError(f"unexpected block number: {num}")

    def get_total_blocks(self) -> int:
        return self._total_blocks

    def _set_blocks(self):
        for i in range(1, 10):
            setattr(self, f"block{i}", self.get_block(i))


def make_blocks(model: SupportedAllModuleType) -> Union[ResnetBlocks, WRNBlocks]:
    if isinstance(model, WideResNet) or isinstance(model, ParsevalWideResNet):
        return WRNBlocks(model)
    elif isinstance(model, ResNet) or isinstance(model, ParsevalResNet):
        return ResnetBlocks(model)
    else:
        raise ValueError(f"model {type(model).__name__} is not supported to divide into blocks")


if __name__ == '__main__':
    from .resnet import resnet18
    from .parseval_resnet import parseval_resnet18
    from .wrn import wrn34_10
    from .parseval_wrn import parseval_retrain_wrn34_10

    # model = resnet18(num_classes=10)
    model = wrn34_10(num_classes=10)
    print(model)
    # blocks = make_blocks(model)

    # print(blocks.block8)