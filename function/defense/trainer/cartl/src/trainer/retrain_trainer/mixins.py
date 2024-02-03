from torch.nn.modules.module import Module
from ....src.networks.utils import make_blocks
from ....src.config import Settings
from typing import List, Tuple, Union

from torch.nn import BatchNorm2d

from ....src.utils import logger
from ....src.networks import WRNBlocks, ResnetBlocks
from ....src.networks import SupportedAllModuleType


class ResetBlockMixin:
    model: SupportedAllModuleType
    _blocks: Union[WRNBlocks, ResnetBlocks]

    def reset_and_unfreeze_last_k_blocks(self, k: int):
        """reset and unfreeze layers in last k blocks

        Args:
            k: the last k blocks which will be retrained
        """
        total_blocks = self._blocks.get_total_blocks()
        logger.debug(f"model: {type(self.model).__name__}, blocks: {total_blocks}")
        for i in range(total_blocks, total_blocks-k, -1):
            block = getattr(self._blocks, f"block{i}")
            for layer in block:
                if hasattr(layer, "reset_parameters"):
                    logger.debug(f"reinitialize layer: {type(layer).__name__}")
                    layer.reset_parameters()
            for p in block.parameters():
                p.requires_grad = True

        logger.debug(f"unfreeze and reset last {k} blocks")
        logger.debug("trainable layers")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.debug(f"name: {name}, size: {param.size()}")

    def reset_last_k_blocks(self, k: int):
        """reset layers in last k blocks

        Args:
            k: the last k blocks which will be retrained
        """
        total_blocks = self._blocks.get_total_blocks()
        logger.debug(f"model: {type(self.model).__name__}, blocks: {total_blocks}")
        for i in range(total_blocks, total_blocks-k, -1):
            block = getattr(self._blocks, f"block{i}")
            for layer in block:
                if hasattr(layer, "reset_parameters"):
                    logger.debug(f"reinitialize layer: {type(layer).__name__}")
                    layer.reset_parameters()

        logger.debug(f"reset last {k} blocks")

    def unfreeze_last_k_blocks(self, k: int):
        """unfreeze layers in last k blocks

        Args:
            k: the last k blocks which will be retrained
        """
        total_blocks = self._blocks.get_total_blocks()
        logger.debug(f"model: {type(self.model).__name__}, blocks: {total_blocks}")
        for i in range(total_blocks, total_blocks-k, -1):
            block = getattr(self._blocks, f"block{i}")
            for p in block.parameters():
                p.requires_grad = True


class FreezeModelMixin:
    """freeze all parameters of model"""
    model: SupportedAllModuleType
    _blocks: Union[ResnetBlocks, WRNBlocks]

    def freeze_model(self):
        for p in self.model.parameters():
            p.requires_grad = False

        logger.debug(f"all parameters of model are freezed")

    def unfreeze_model(self):
        for p in self.model.parameters():
            p.requires_grad = True

        logger.debug(f"all parameters of model are unfreezed")
    
    def _freeze_bn_proc(self, layer:Module, reuse_statistic):
        for p in layer.parameters():
            p.requires_grad = False
        if reuse_statistic:
            layer.eval()


    def freeze_bn_layer(self, verbose=True, freezing_range:Union[Tuple[int, int],None]=None, reuse_statistic=False):
        """
            verbose: print log.
            freezing_range: indicating range where batch norm layer to be freezed.
                            A tuple (start, end) or a list [1, 2, ..] can be provided.
            reuse_statistic: using previous statistic (running_mean and running_var).
        """
        # Different from previous version, we would freezing batch norm layers in unit of block.
        # Thus, layers which are NOT in the 'block k' are not considerred, e.g., self.conv1 for WRN 34-10.
        # And WE ASSUME THOSE LATERS ARE NOT BATCH NORM LAYERS.
        # -- imTyrant

        # difference between modules and parameters
        # https://blog.paperspace.com/pytorch-101-advanced/
        assert hasattr(self, "_blocks")
        if isinstance(self._blocks, ResnetBlocks):
            # TODO hard...
            logger.warning( "*" * 100 + 
                            "\nResNet Detected."
                            "\nIn current code, it is hard to freeze the batch norm layer in the 'self.conv1'."
                            "\nI will manually freeze it!\n" +
                            "*" * 100)
            for layer in self.model.conv1.modules():
                self._freeze_bn_proc(layer, reuse_statistic)


        if freezing_range is None:
            freezing_range = list(range(1, self._blocks.get_total_blocks() + 1))
        else:
            assert (isinstance(freezing_range, tuple) and len(freezing_range) == 2) \
                    or isinstance(freezing_range, list)
            if isinstance(freezing_range, tuple):
                if freezing_range[1] == -1:
                    _upper_range = self._blocks.get_total_blocks()
                else:
                    _upper_range = min(self._blocks.get_total_blocks(), freezing_range[1])
                _low_range = max(1, freezing_range[0])
                freezing_range = list(range(_low_range, _upper_range + 1))
            elif isinstance(freezing_range, list):
                # TODO: efficiently check validity of each element
                pass
            else:
                raise ValueError("'freezing_range' must be a two-element tuple or a list.")
        
        assert isinstance(freezing_range, list)
        for bi in freezing_range:
            block = getattr(self._blocks, f"block{bi}")
            for layer in block.modules():
                if isinstance(layer, BatchNorm2d):
                    self._freeze_bn_proc(layer, reuse_statistic)
                    
        if verbose:
            logger.debug(f"batch norm layers in block{freezing_range} are freezed")
            if reuse_statistic:
                logger.debug("also using teacher's running statistics")