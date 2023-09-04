from ...src import settings
from ...src.utils import logger

import os

from torch.utils.tensorboard import SummaryWriter


class InitializeTensorboardMixin:
    """provide some useful function in tensorboard"""

    _checkpoint_path: str

    def init_writer(self, log_dir: str = None) -> SummaryWriter:
        """initialize `SummaryWriter`, this should be called after super().__init__

        Args:
            log_dir: specify log_dir of SummaryWriter, if the value if None,
                     it will be specified as `{settings.tensorboard_log_dir}/{filename of checkpoint path}`
        """
        if not log_dir:
            checkpoint_name = os.path.splitext(self._checkpoint_path.split("/")[-1])[0]
            log_dir = settings.tensorboard_log_dir / checkpoint_name
        logger.info(f"tensorboard log dir: {log_dir}")

        return SummaryWriter(log_dir=log_dir)
