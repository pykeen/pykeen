"""PyKEEN is a Python package for reproducible, facile knowledge graph embeddings.

The fastest way to get up and running is to use the :func:`pykeen.pipeline.pipeline`
function.

It provides a high-level entry into the extensible functionality of
this package. The following example shows how to train and evaluate the
TransE model (:class:`pykeen.models.TransE`) on the Nations dataset (:class:`pykeen.datasets.Nations`)
by referring to them by name. By default, the training loop uses the stochastic closed world assumption training
approach
(:class:`pykeen.training.SLCWATrainingLoop`) and evaluates with rank-based evaluation
(:class:`pykeen.evaluation.RankBasedEvaluator`).

>>> from pykeen.pipeline import pipeline
>>> result = pipeline(
...     model='TransE',
...     dataset='Nations',
... )

The results are returned in a :class:`pykeen.pipeline.PipelineResult` instance, which has
attributes for the trained model, the training loop, and the evaluation.

PyKEEN has a function :func:`pykeen.env` that magically prints relevant version information
about PyTorch, CUDA, and your operating system that can be used for debugging.
If you're in a Jupyter notebook, it will be pretty printed as an HTML table.

>>> import pykeen
>>> pykeen.env()
"""

import logging

from .version import env, get_git_branch, get_git_hash, get_version  # noqa: F401

# This will set the global logging level to info to ensure that info messages are shown in all parts of the software.
logging.getLogger(__name__).setLevel(logging.INFO)
