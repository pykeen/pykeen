# -*- coding: utf-8 -*-

"""POEM is a Python package for reproducible, facile knowledge graph embeddings.

The fastest way to get up and running is to use the :func:`poem.pipeline.pipeline`
function.

It provides a high-level entry into the extensible functionality of
this package. The following example shows how to train and evaluate the
TransE model (:class:`poem.models.TransE`) on the Nations dataset (:class:`poem.datasets.Nations`)
by referring to them by name. By default, the training loop uses the open world assumption
(:class:`poem.training.OWATrainingLoop`) and evaluates with rank-based evaluation
(:class:`poem.evaluation.RankBasedEvaluator`).

>>> from poem.pipeline import pipeline
>>> result = pipeline(
...     model='TransE',
...     data_set='Nations',
... )

The results are returned in a :class:`poem.pipeline.PipelineResult` instance, which has
attributes for the trained model, the training loop, and the evaluation.
"""

import logging

from .version import get_version  # noqa: F401

# This will set the global logging level to info to ensure that info messages are shown in all parts of the software.
logging.getLogger(__name__).setLevel(logging.INFO)
