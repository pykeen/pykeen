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

from typing import Collection, Mapping, Type, Union

from .early_stopping import EarlyStopper, StopperCallback  # noqa: F401
from .stopper import NopStopper, Stopper
from ..utils import get_cls, normalize_string

__all__ = [
    'Stopper',
    'NopStopper',
    'EarlyStopper',
    'get_stopper_cls',
]

_STOPPER_SUFFIX = 'Stopper'
_STOPPERS: Collection[Type[Stopper]] = {
    NopStopper,
    EarlyStopper,
}

#: A mapping of stoppers' names to their implementations
stoppers: Mapping[str, Type[Stopper]] = {
    normalize_string(cls.__name__, suffix=_STOPPER_SUFFIX): cls
    for cls in _STOPPERS
}


def get_stopper_cls(query: Union[None, str, Type[Stopper]]) -> Type[Stopper]:
    """Look up a stopper class by name (case/punctuation insensitive) in :data:`pykeen.stoppers.stoppers`.

    :param query: The name of the stopper (case insensitive, punctuation insensitive).
    :return: The stopper class
    """
    return get_cls(
        query,
        base=Stopper,
        lookup_dict=stoppers,
        default=NopStopper,
        suffix=_STOPPER_SUFFIX,
    )
