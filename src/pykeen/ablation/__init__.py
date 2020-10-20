# -*- coding: utf-8 -*-

"""Ablation studies in PyKEEN.

An ablation study is an experiment in which components of a machine learning system are removed/replaced in order
to measure the impact of these components on the system's performance.
In PyKEEN, a user could for instance measure the impact of explicitly modeling inverse relations on the model's
performance. This can be done with the :func:`pykeen.ablation.ablation_pipeline` function.

.. code-block:: python

    from pykeen.ablation import ablation_pipeline

    ablation_result = ablation_pipeline(
        datasets='kinships',
        models=['RotatE', 'TransE'],
        losses=['BCEAfterSigmoidLoss', 'NSSA'],
        optimizers='Adam',
        training_loops=['sLCWA', 'LCWA'],
        optuna_config={
            'n_trials': 5,
        },
        directory='~/Desktop/simple_ablation_study',
    )

.. note:: This tutorial will be improved with `pykeen/pykeen#116 <https://github.com/pykeen/pykeen/issues/116>`_.
"""

from .ablation import (
    ablation_pipeline, ablation_pipeline_from_config, prepare_ablation, prepare_ablation_from_config,
    prepare_ablation_from_path,
)

__all__ = [
    'ablation_pipeline',
    'ablation_pipeline_from_config',
    'prepare_ablation_from_config',
    'prepare_ablation_from_path',
    'prepare_ablation',
]
