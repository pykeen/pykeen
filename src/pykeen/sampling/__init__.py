# -*- coding: utf-8 -*-

r"""For entities $\mathcal{E}$ and relations $\mathcal{R}$, the set of all possible triples $\mathcal{T}$ is 
constructed through their cartesian product $\mathcal{T} = \mathcal{E} \times \mathcal{R} \times \mathcal{E}$.
A given knowledge graph $\mathcal{K}$ is a subset of all possible triples $\mathcal{K} \subseteq \mathcal{T}$.

Construction of Knowledge Graphs
--------------------------------
When constructing a knowledge graph $\mathcal{K}_{\text{closed}}$ under the closed world assumption, the labels of the 
remaining triples $(h,r,t) \in \mathcal{T} \setminus \mathcal{K}_{\text{closed}}$ are defined as negative. 
When constructing a knowledge graph $\mathcal{K}_{\text{open}}$ under the open world assumption, the labels of the
remaining triples $(h,r,t) \in \mathcal{T} \setminus \mathcal{K}_{\text{open}}$ are unknown.

Becuase most knowledge graphs are generated under the open world assumption, negative sampling techniques
must be employed during the training of knowledge graph embedding models to avoid over-generalization.

Corruption
----------
Negative sampling techniques often generate negative triples by corrupting a known positive triple
$(h,r,t) \in \mathcal{K}$ by replacing either $h$, $r$, or $t$ with one of the following operations:

=================  =====================================================================================
Corrupt heads      :math:`\mathcal{H}(h, r, t) = \{(h', r, t) \mid h' \in \mathcal{E} \land h' \neq h\}`
Corrupt relations  :math:`\mathcal{R}(h, r, t) = \{(h, r', t) \mid r' \in \mathcal{E} \land r' \neq r\}`
Corrupt tails      :math:`\mathcal{T}(h, r, t) = \{(h, r, t') \mid t' \in \mathcal{E} \land t' \neq t\}`
=================  =====================================================================================

Typically, the corrupt relations operation $\mathcal{R}(h, r, t)$ is omitted becuase the evaluation of knowledge
graph embedding models on the link prediction task only consideres the goodness of head prediction and tail
prediction, but not relation prediction. Therefore, the set of candidate negative triples $\mathcal{N}(h, r, t)$ for
a given known positive triple $(h,r,t) \in \mathcal{K}$ is given by:

.. math::

    \mathcal{N}(h, r, t) = \mathcal{T}(h, r, t) \cup \mathcal{H}(h, r, t)

Generally, the set of potential negative triples $\mathcal{N}$ over all positive triples $(h,r,t) \in \mathcal{K}$
is defined as:

.. math::

    \mathcal{N} = \bigcup_{(h,r,t) \in \mathcal{K}} \mathcal{N}(h, r, t)

Uniform Negative Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~
The default negative sampler :class:`pykeen.sampling.BasicNegativeSampler` generates corrupted triples from
a known positive triple $(h,r,t) \in \mathcal{K}$ by uniformly randomly either using the corrupt heads operation
or the corrupt tails operation. The default negative sampler is automatically used in the following code:

.. code-block:: python

    from pykeen.pipeline import pipeline

    results = pipeline(
        dataset='YAGO3-10',
        model='PairRE',
        training_loop='sLCWA',
    )


It can be set explicitly with:

.. code-block:: python

    from pykeen.pipeline import pipeline

    results = pipeline(
        dataset='YAGO3-10',
        model='PairRE',
        training_loop='sLCWA',
        negative_sampler='basic',
    )
    
In general, the behavior of the negative sampler can be modified when using the :func:`pykeen.pipeline.pipeline` by
passing the ``negative_sampler_kwargs`` argument. In order to explicitly specifiy which of the head, relation, and
tail corruption methods are used, the ``corruption_schema`` argument can be used. For example, to use all three,
the collection ``('h', 'r', 't')`` can be passed as in the following:

.. code-block:: python

    from pykeen.pipeline import pipeline

    results = pipeline(
        dataset='YAGO3-10',
        model='PairRE',
        training_loop='sLCWA',
        negative_sampler='basic',
        negative_sampler_kwargs=dict(
            corruption_scheme=('h', 'r', 't'),
        ),
    )

Bernoulli Negative Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Bernoulli negative sampler :class:`pykeen.sampling.BernoulliNegativeSampler` generates corrupted triples from
a known positive triple $(h,r,t) \in \mathcal{K}$ similarly to the uniform negative sampler, but it pre-computes
a probability $p_r$ for each relation $r$ to weight whether the head corruption is used with probability $p_r$ or
if tail corruption is used with probability $1 - p_r$.

.. code-block:: python

    from pykeen.pipeline import pipeline

    results = pipeline(
        dataset='YAGO3-10',
        model='PairRE',
        training_loop='sLCWA',
        negative_sampler='bernoulli',
    )
"""  # noqa

from class_resolver import ClassResolver

from .basic_negative_sampler import BasicNegativeSampler
from .bernoulli_negative_sampler import BernoulliNegativeSampler
from .negative_sampler import NegativeSampler
from .pseudo_type import PseudoTypedNegativeSampler

__all__ = [
    "NegativeSampler",
    "BasicNegativeSampler",
    "BernoulliNegativeSampler",
    "PseudoTypedNegativeSampler",
    # Utils
    "negative_sampler_resolver",
]

negative_sampler_resolver: ClassResolver[NegativeSampler] = ClassResolver.from_subclasses(
    NegativeSampler,
    default=BasicNegativeSampler,
)
