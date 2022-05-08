# -*- coding: utf-8 -*-

"""Early stoppers.

The following code will create a scenario in which training will stop
(quite) early when training :class:`pykeen.models.TransE` on the
:class:`pykeen.datasets.Nations` dataset.

>>> from pykeen.pipeline import pipeline
>>> pipeline_result = pipeline(
...     dataset='nations',
...     model='transe',
...     model_kwargs=dict(embedding_dim=20, scoring_fct_norm=1),
...     optimizer='SGD',
...     optimizer_kwargs=dict(lr=0.01),
...     loss='marginranking',
...     loss_kwargs=dict(margin=1),
...     training_loop='slcwa',
...     training_kwargs=dict(num_epochs=100, batch_size=128),
...     negative_sampler='basic',
...     negative_sampler_kwargs=dict(num_negs_per_pos=1),
...     evaluator_kwargs=dict(filtered=True),
...     evaluation_kwargs=dict(batch_size=128),
...     stopper='early',
...     stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
... )
"""

from class_resolver import ClassResolver

from .early_stopping import EarlyStopper, StopperCallback  # noqa: F401
from .stopper import NopStopper, Stopper

__all__ = [
    "Stopper",
    "NopStopper",
    "EarlyStopper",
    # Utils
    "stopper_resolver",
]

stopper_resolver: ClassResolver[Stopper] = ClassResolver.from_subclasses(
    Stopper,
    default=NopStopper,
)
