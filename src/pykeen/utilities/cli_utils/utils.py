# -*- coding: utf-8 -*-

"""CLI utils."""

from collections import OrderedDict

from pykeen.constants import KG_EMBEDDING_MODEL_NAME


def get_config_dict(model_name):
    """Get configuration dictionary.

    :param str model_name:
    :rtype: dict
    """
    config = OrderedDict()
    config[KG_EMBEDDING_MODEL_NAME] = model_name
    return config
