# -*- coding: utf-8 -*-

"""CLI utils."""

import json
import os
from collections import OrderedDict
import pandas as pd
import click
from pykeen.constants import KG_EMBEDDING_MODEL_NAME


def get_config_dict(model_name):
    """Get configuration dictionary.

    :param str model_name:
    :rtype: dict
    """
    config = OrderedDict()
    config[KG_EMBEDDING_MODEL_NAME] = model_name
    return config


def summarize_results(directory: str, output):
    """Summarize contents of training and evaluation"""
    r = []
    for subdirectory_name in os.listdir(directory):
        subdirectory = os.path.join(directory, subdirectory_name)
        if not os.path.isdir(subdirectory):
            continue
        configuration_path = os.path.join(subdirectory, 'configuration.json')
        if not os.path.exists(configuration_path):
            click.echo("missing configuration")
            continue
        with open(configuration_path) as file:
            configuration = json.load(file)
        evaluation_path = os.path.join(subdirectory, 'evaluation_summary.json')
        if not os.path.exists(evaluation_path):
            click.echo("missing evaluation summary")
            continue
        with open(evaluation_path) as file:
            evaluation = json.load(file)
        r.append(dict(**configuration, **evaluation))
    df = pd.DataFrame(r)
    df.to_csv(output, sep='\t')
