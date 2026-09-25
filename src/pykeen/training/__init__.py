r"""Training loops for KGE models using multi-modal information.

Throughout the following explanations of training loops, we will assume the
set of entities $\mathcal{E}$, set of relations $\mathcal{R}$,
set of possible triples $\mathcal{T} = \mathcal{E} \times \mathcal{R} \times \mathcal{E}$.
We stratify $\mathcal{T}$ into the `disjoint union <https://en.wikipedia.org/wiki/Disjoint_union>`_
of positive triples $\mathcal{T^{+}} \subseteq \mathcal{T}$ and
negative triples $\mathcal{T^{-}} \subseteq \mathcal{T}$
such that $\mathcal{T^{+}} \cap \mathcal{T^{-}} = \emptyset$
and $\mathcal{T^{+}} \cup \mathcal{T^{-}} = \mathcal{T}$.

A knowledge graph $\mathcal{K}$ constructed under the open world assumption contains a subset
of all possible positive triples such that $\mathcal{K} \subseteq \mathcal{T^{+}}$.

Assumptions
-----------

Open World Assumption
~~~~~~~~~~~~~~~~~~~~~
When training under the open world assumption (OWA), all triples that are not part of the
knowledge graph are considered unknown (e.g., neither positive nor negative).
This leads to under-fitting (i.e., over-generalization) and is therefore usually a poor choice for
training knowledge graph embedding models [nickel2016review]_. PyKEEN does *not* implement a training loop
with the OWA.

.. warning::

    Many publications and software packages use OWA to incorrectly refer to the stochastic
    local closed world assumption (sLCWA). See below for an explanation.

Closed World Assumption
~~~~~~~~~~~~~~~~~~~~~~~
When training under the close world assumption (CWA), all triples that are not part of the
knowledge graph are considered as negative. As most knowledge graphs are inherently incomplete,
this leads to over-fitting and is therefore usually a poor choice for training knowledge
graph embedding models. PyKEEN does *not* implement a training loop with the CWA.

Local Closed World Assumption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When training under the local closed world assumption (LCWA; introduced in [dong2014]_),
a particular subset of triples that are not part of the knowledge graph are considered as
negative.

===========  =================================================================================================  ================================================================
Strategy     Local Generator                                                                                    Global Generator
===========  =================================================================================================  ================================================================
Head         $\mathcal{T}_h^-(r,t)=\{(h,r,t) \mid h \in \mathcal{E} \land (h,r,t) \notin \mathcal{K} \}$        $\bigcup\limits_{(\_,r,t) \in \mathcal{K}} \mathcal{T}_h^-(r,t)$
Relation     $\mathcal{T}_r^-(h,t)=\{(h,r,t) \mid r \in \mathcal{R} \land (h,r,t) \notin \mathcal{K} \}$        $\bigcup\limits_{(h,\_,t) \in \mathcal{K}} \mathcal{T}_r^-(h,t)$
Tail         $\mathcal{T}_t^-(h,r)=\{(h,r,t) \mid t \in \mathcal{E} \land (h,r,t) \notin \mathcal{K} \}$        $\bigcup\limits_{(h,r,\_) \in \mathcal{K}} \mathcal{T}_t^-(h,r)$
===========  =================================================================================================  ================================================================

Most articles refer exclusively to the tail generation strategy when discussing LCWA. However, the relation
generation strategy is a popular choice in visual relation detection domain (see [zhang2017]_ and
[sharifzadeh2019vrd]_). However, PyKEEN additionally implements head generation since
`PR #602 <https://github.com/pykeen/pykeen/pull/602>`_.

Stochastic Local Closed World Assumption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When training under the stochastic local closed world assumption (SLCWA), a random subset of the union of
the head and tail generation strategies from LCWA are considered as negative triples. There are a few
benefits from doing this:

1. Reduce computational workload
2. Spare updates (i.e., only a few rows of the embedding are affected)
3. Ability to integrate new negative sampling strategies

There are two other major considerations when randomly sampling negative triples: the random sampling
strategy and the filtering of positive triples. A full guide on negative sampling with the SLCWA can be
found in :mod:`pykeen.sampling`. The following chart from [ali2020a]_ demonstrates the different potential
triples considered in LCWA vs. sLCWA based on the given true triples (in red):

.. image:: ../img/training_approaches.png
  :alt: Troubleshooting Image 2

Batch-Local Closed World Assumption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When training under the batch-local closed world assumption (BCWA), all combinations of heads, relations, and tails
occurring in a batch of training triples are scored, and those which are not training triples are considered as
negative. See :class:`~pykeen.training.BatchCWATrainingLoop` for details.

For example, consider a knowledge graph with the training triples (Alice, knows, Bob), (Bob, likes, Carol),
(Alice, likes, Carol), and (Carol, knows, Dave), and a batch with the first two triples. The batch contains the heads
{Alice, Bob}, the relations {knows, likes}, and the tails {Bob, Carol}, and BCWA scores all $2 \cdot 2 \cdot 2 = 8$
combinations of them. Three of them are training triples and thus positive: the two batch triples, and
(Alice, likes, Carol), which is not part of the batch. The remaining five are negative. For tail prediction, each
(head, relation)-pair is one row, and the batch's tails are the candidates:

================  ===  =====
(head, relation)  Bob  Carol
================  ===  =====
(Alice, knows)    1    0
(Alice, likes)    0    1
(Bob, knows)      0    0
(Bob, likes)      0    1
================  ===  =====

In comparison, the CWA would score all $4 \cdot 2 \cdot 4 = 32$ possible triples, and the LCWA would score each
(head, relation)-pair of the batch with all four entities as tails. Under the BCWA, Dave is never scored, since he
does not occur in the batch. Hence, it only requires the representations of the batch's entities and relations, which
are needed to score the batch triples anyway.
"""  # noqa:E501

from class_resolver import ClassResolver

from .bcwa import BatchCWATrainingLoop  # noqa: F401
from .callbacks import TrainingCallback, callback_resolver  # noqa: F401
from .lcwa import LCWATrainingLoop, SymmetricLCWATrainingLoop  # noqa: F401
from .slcwa import SLCWATrainingLoop  # noqa: F401
from .training_loop import (  # noqa: F401
    NonFiniteLossError,
    OptimizerClearedError,
    OptimizerNotRecreatableError,
    TrainingLoop,
)

__all__ = [
    "TrainingLoop",
    "SLCWATrainingLoop",
    "LCWATrainingLoop",
    "SymmetricLCWATrainingLoop",
    "BatchCWATrainingLoop",
    "NonFiniteLossError",
    "OptimizerNotRecreatableError",
    "OptimizerClearedError",
    "training_loop_resolver",
    #
    "TrainingCallback",
    "callback_resolver",
]

#: A resolver for training loops
training_loop_resolver: ClassResolver[TrainingLoop] = ClassResolver.from_subclasses(
    base=TrainingLoop,  # type: ignore[type-abstract]
    default=SLCWATrainingLoop,
)
