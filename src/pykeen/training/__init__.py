# -*- coding: utf-8 -*-

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
"""  # noqa:E501

from class_resolver import ClassResolver

from .callbacks import TrainingCallback  # noqa: F401
from .lcwa import LCWATrainingLoop  # noqa: F401
from .slcwa import SLCWATrainingLoop  # noqa: F401
from .training_loop import NonFiniteLossError, TrainingLoop  # noqa: F401

__all__ = [
    "TrainingLoop",
    "SLCWATrainingLoop",
    "LCWATrainingLoop",
    "NonFiniteLossError",
    "training_loop_resolver",
    "TrainingCallback",
]

training_loop_resolver: ClassResolver[TrainingLoop] = ClassResolver.from_subclasses(
    base=TrainingLoop,  # type: ignore
    default=SLCWATrainingLoop,
)
