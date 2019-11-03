# -*- coding: utf-8 -*-

"""A collection of default experiments.

A wrapper around an experiment with TransE and the Nations
data set can be found at :func:`poem.experiments.run_trans_e_experiment`.

This example shows how it can be run to make 6 replicates, then plot the
losses together.

>>> from poem.experiments import run_example_experiment
>>> result = run_example_experiment(replicates=6)
>>> ax = result.plot_losses()
>>> fig = ax.get_figure()
>>> fig.savefig('losses.png')
"""

import os
from typing import Optional

from .utils import PipelineResultSet, pipeline_from_path
from ..pipeline import PipelineResult

HERE = os.path.abspath(os.path.dirname(__file__))


def run_example_experiment(replicates: Optional[int] = None):
    """Run the TransE experiments."""
    path = os.path.join(HERE, 'example_experiment.json')
    if replicates is None:
        return pipeline_from_path(path)
    return PipelineResultSet.from_path(path, replicates=replicates)


def run_bordes2013_transe_wn18() -> PipelineResult:
    """Train TransE on WordNet as described in [bordes2013]_."""
    path = os.path.join(HERE, 'trans_e', 'bordes2013_transe_wn18.json')
    return pipeline_from_path(path)


def run_bordes2013_transe_fb15k() -> PipelineResult:
    """Train TransE on FB15K as described in [bordes2013]_."""
    path = os.path.join(HERE, 'trans_e', 'bordes2013_transe_fb15k.json')
    return pipeline_from_path(path)
