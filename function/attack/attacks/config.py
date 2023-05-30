import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------- CONSTANTS AND TYPES

MY_NUMPY_DTYPE = np.float32  # pylint: disable=C0103
MY_DATA_PATH: str

# --------------------------------------------------------------------------------------------- DEFAULT PACKAGE CONFIGS

_folder = os.path.expanduser("~")
if not os.access(_folder, os.W_OK):  # pragma: no cover
    _folder = "/tmp"  # pylint: disable=C0103
_folder = os.path.join(_folder, ".Attack")


def set_data_path(path):
    """
    Set the path for Attack's data directory (MY_DATA_PATH).
    """
    expanded_path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(expanded_path, exist_ok=True)
    if not os.access(expanded_path, os.R_OK):  # pragma: no cover
        raise OSError(f"path {expanded_path} cannot be read from")
    if not os.access(expanded_path, os.W_OK):  # pragma: no cover
        logger.warning("path %s is read only", expanded_path)

    global MY_DATA_PATH  # pylint: disable=W0603
    MY_DATA_PATH = expanded_path
    logger.info("set MY_DATA_PATH to %s", expanded_path)


# Load data from configuration file if it exists. Otherwise create one.
_config_path = os.path.expanduser(os.path.join(_folder, "config.json"))
if os.path.exists(_config_path):
    try:
        with open(_config_path, encoding="utf8") as f:
            _config = json.load(f)

            # Since renaming this variable we must update existing config files
            if "DATA_PATH" in _config:  # pragma: no cover
                _config["MY_DATA_PATH"] = _config.pop("DATA_PATH")
                try:
                    with open(_config_path, "w", encoding="utf8") as f:
                        f.write(json.dumps(_config, indent=4))
                except IOError:
                    logger.warning("Unable to update configuration file", exc_info=True)

    except ValueError:  # pragma: no cover
        _config = {}

if not os.path.exists(_folder):
    try:
        os.makedirs(_folder)
    except OSError:  # pragma: no cover
        logger.warning("Unable to create folder for configuration file.", exc_info=True)

if not os.path.exists(_config_path):
    # Generate default config
    _config = {"MY_DATA_PATH": os.path.join(_folder, "data")}

    try:
        with open(_config_path, "w", encoding="utf8") as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:  # pragma: no cover
        logger.warning("Unable to create configuration file", exc_info=True)

if "MY_DATA_PATH" in _config:  # pragma: no cover
    set_data_path(_config["MY_DATA_PATH"])
