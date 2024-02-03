import toml

import logging.config
from typing import Union, Optional, Callable
from pathlib import PurePath
import functools

from ...src import settings


class MyLogger:
    """wrap of logging in standard library, support reset log file after initialization"""

    class _Decorators:
        @staticmethod
        def check_logger_initialized(func: Callable):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                if not self._logger:
                    raise AttributeError(f"Before using method `{func.__name__}`, "
                                         f"initialize logger with `set_logger`!")
                return func(self, *args, **kwargs)

            return wrapper

    def __init__(self, config_path: Union[str, PurePath]):
        with open(config_path, "r", encoding="utf8") as f:
            config_dict = toml.loads(f.read())
        self._config_dict = config_dict

        self._logger: Optional[logging.Logger] = None

    @_Decorators.check_logger_initialized
    def info(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'INFO'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
        """
        self._logger.info(msg, *args, **kwargs)

    @_Decorators.check_logger_initialized
    def debug(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'DEBUG'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.debug("Houston, we have a %s", "thorny problem", exc_info=1)
        """
        self._logger.debug(msg, *args, **kwargs)

    @_Decorators.check_logger_initialized
    def warning(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        self._logger.warning(msg, *args, **kwargs)

    def set_log_file(self, filename: Union[str, PurePath]):
        self._config_dict["handlers"]["file"]["filename"] = filename

    def set_logger(self, logger_name: str = settings.logger_name):
        assert logger_name in {"StreamLogger", "FileLogger"}

        if not self._config_dict["handlers"]["file"].get("filename"):
            raise ValueError("cal method `set_log_file` to set log file!")

        self._clean_handlers()

        logging.config.dictConfig(self._config_dict)
        self._logger = logging.getLogger(logger_name)

    def change_log_file(self, filename: Union[str, PurePath]):
        """equals `set_log_file + set_logger`"""
        self.set_log_file(filename=filename)
        self.set_logger(logger_name="FileLogger")

    def _clean_handlers(self):
        if isinstance(self._logger, logging.Logger):
            for handles in self._logger.handlers:
                handles.close()
                self._logger.removeHandler(handles)


logger = MyLogger(settings.logger_config_file)
