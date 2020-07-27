# -*- coding: utf-8 -*-

"""Ablation studies in PyKEEN.

  "If :func:`pykeen.pipeline.pipeline` were making a maze, :func:`pykeen.hpo.hpo_pipeline` were getting
  killed by your dead wife, then :func:`pykeen.ablation.prepare_ablation_from_config` is washing up on
  a beach." -Christopher Nolan

TODO for Mehdi!

1. What is an ablation study?
2. Why would a user want to run an ablation study?
   1. Give example from benchmarking paper
3. How can a user run an ablation study?
   1. Minimal Example (written in python code embedded in this RST document)
   2. More complicated usage, step-by-step. See the Pipeline Tutorial as an example on how this might look

.. warning::

  do not use JSON files for examples. All usage should be through the python dictionary interface. If we
  forgot to make a python dictionary interface, then we must add one for this PR

"""

from .ablation import prepare_ablation, prepare_ablation_from_config

__all__ = [
    'prepare_ablation_from_config',
    'prepare_ablation',
]
