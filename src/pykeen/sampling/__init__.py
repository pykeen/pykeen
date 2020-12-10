# -*- coding: utf-8 -*-

r"""Because most knowledge graphs are generated under the open world assumption, knowledge
graph embedding models must be trained involving techniques such as negative sampling to
avoid over-generalization.

Two common approaches for generating negative samples are :class:`pykeen.sampling.BasicNegativeSampler`
and :class:`pykeen.sampling.BernoulliBasicSampler` in which negative triples are created by corrupting
a positive triple $(h,r,t) \in \mathcal{K}$ by replacing either $h$ or $t$.
We denote with $\mathcal{N}$ the set of all potential negative triples:

.. math::
    \mathcal{N} &=& \bigcup_{(h,r,t) \in \mathcal{K}} \mathcal{N}(h, r, t)\\
    \mathcal{N}(h, r, t) &=& \mathcal{T}(h, r) \cup \mathcal{H}(r, t)\\
    \mathcal{T}(h, r) &=& \{(h, r, t') \mid t' \in \mathcal{E} \land t' \neq t\}\\
    \mathcal{H}(r, t) &=& \{(h', r, t) \mid h' \in \mathcal{E} \land h' \neq h\}

In theory, all positive triples in $\mathcal{K}$ should be excluded from this set of candidate negative
triples $\mathcal{N}$ such that $\mathcal{N}^- = \mathcal{N} \setminus \mathcal{K}$. In practice, however,
since usually $|\mathcal{N}| \gg |\mathcal{K}|$, the likelihood of generating a false negative is rather low.
Therefore, the additional filter step is often omitted to lower computational cost. It should be taken
into account that a corrupted triple that is *not part* of the knowledge graph can represent a true fact.
"""  # noqa

from typing import Mapping, Set, Type, Union

from .basic_negative_sampler import BasicNegativeSampler
from .bernoulli_negative_sampler import BernoulliNegativeSampler
from .negative_sampler import NegativeSampler
from ..utils import get_cls, normalize_string

__all__ = [
    'NegativeSampler',
    'BasicNegativeSampler',
    'BernoulliNegativeSampler',
    'negative_samplers',
    'get_negative_sampler_cls',
]

_NEGATIVE_SAMPLER_SUFFIX = 'NegativeSampler'
_NEGATIVE_SAMPLERS: Set[Type[NegativeSampler]] = {
    BasicNegativeSampler,
    BernoulliNegativeSampler,
}

#: A mapping of negative samplers' names to their implementations
negative_samplers: Mapping[str, Type[NegativeSampler]] = {
    normalize_string(cls.__name__, suffix=_NEGATIVE_SAMPLER_SUFFIX): cls
    for cls in _NEGATIVE_SAMPLERS
}


def get_negative_sampler_cls(query: Union[None, str, Type[NegativeSampler]]) -> Type[NegativeSampler]:
    """Get the negative sampler class."""
    return get_cls(
        query,
        base=NegativeSampler,
        lookup_dict=negative_samplers,
        default=BasicNegativeSampler,
        suffix=_NEGATIVE_SAMPLER_SUFFIX,
    )
