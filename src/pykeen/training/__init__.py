# -*- coding: utf-8 -*-

r"""Training loops for KGE models using multi-modal information.

Throughout the following explanations of training loops, we will assume the
set of entities $\mathcal{E}$, set of relations $\mathcal{R}$,
set of possible triples $\mathcal{T} = \mathcal{E} \times \mathcal{R} \times \mathcal{E}$.
We stratify $\mathcal{T}$ into the `disjoint union <https://en.wikipedia.org/wiki/Disjoint_union>`_
of positive triples $\mathcal{K} \subseteq \mathcal{T}$ and
negative triples $\mathcal{\bar{K}} \subseteq \mathcal{T}$
such that $\mathcal{K} \cap \mathcal{\bar{K}} = \emptyset$
and $\mathcal{K} \cup \mathcal{\bar{K}} = \mathcal{T}$.

Most knowledge graphs are built under the open world assumption (OWA)
and do not contain negative triples. If you trained strictly under the OWA,
then the model would overgeneralize [nickel2016review]_. Therefore, it makes
sense to sample some negative triples., which is the stochastic local closed world
assumption (sLCWA). Many other publications and software packages mistakenly call
training under the sLCWA as OWA.

.. image:: ../img/training_approaches.png
  :alt: Troubleshooting Image 2

Open World Assumption
---------------------
When training under the open world assumption (OWA), all triples that are not part of the
knowledge graph are considered unknown (e.g., neither positive nor negative).
This leads to under-fitting (i.e., over-generalization) and is therefore usually a poor choice for
training knowledge graph embedding models [nickel2016review]_. PyKEEN does _not_ implement a training loop
with the OWA.

.. warning::

    Many publications and software packages use OWA to incorrectly refer to the stochastic
    local closed world assumption (sLCWA). See below for an explanation.

Closed World Assumption
-----------------------
When training under the close world assumption (CWA), all triples that are not part of the
knowledge graph are considered as negative. As most knowledge graphs are inherently incomplete,
this leads to over-fitting and is therefore usually a poor choice for training knowledge
graph embedding models. PyKEEN does *not* implement a training loop with the CWA.

Local Closed World Assumption
-----------------------------
When training under the local closed world assumption (LCWA; introduced in [dong2014]_),
a particular subset of triples that are not part of the knowledge graph are considered as
negative.

In this setting, for any triple $(h,r,t) \in \mathcal{K}$ that has been observed, a set
$\mathcal{T}^-(h,r)$ of negative examples is created by considering all triples
$(h, r, t_i) \notin \mathcal{K}$ as false. Therefore, for our exemplary
\ac{kg} (Figure~\ref{fig:exemplary_kg}) for the pair \textit{(Peter, works\_at)}, the triple
\textit{(Peter, works\_at, DHL)} is a false fact since for this pair only the triple
\textit{(Peter, works\_at, Deutsche Bank)} is part of the \ac{kg}.
Similarly, we can construct $\mathcal{H}^-(r,t)$ based on all triples
$(h_i, r, t) \notin \mathcal{K}$, or $\mathcal{R}^-(h,t)$ based on the
triples $(h, r_i, t) \notin \mathcal{K}$. Constructing $\mathcal{R}^-(h,t)$ is a
popular choice in visual relation detection domain~\cite{zhang2017visual,sharifzadeh2019improving}.
However, most of the works in knowledge graph modeling construct only $\mathcal{T}^-(h, r)$ as
the set of negative examples, and in the context of this work refer to $\mathcal{T}^-(h, r)$ as
the set of negatives examples when speaking about LCWA.


Stochastic Local Closed World Assumption
----------------------------------------
Under the \acf{slcwa}, instead of considering all possible triples
$(h,r,t_i) \notin \mathcal{K}$, $(h_i,r,t) \notin \mathcal{K}$ or $(h,r_i,t) \notin \mathcal{K}$ as false,
we randomly take samples of these sets.

Two common approaches for generating negative samples are \ac{uns}~\cite{Bordes2013} and \ac{bns}~\cite{Wang2014} in
which negative triples are created by corrupting a positive triple $(h,r,t) \in \mathcal{K}$
by replacing either $h$ or $t$.
We use $\mathcal{N}$ to denote the set of all potential negative triples:

.. math::

    \mathcal{T}(h, r) &=& \{(h, r, t') \mid t' \in \mathcal{E} \land t' \neq t\}\\
    \mathcal{H}(r, t) &=& \{(h', r, t) \mid h' \in \mathcal{E} \land h' \neq h\}\\
    %\mathcal{N}(h, r, t) &=& \mathcal{T}(h, r) \cup \mathcal{H}(r, t)\\
    \mathcal{N} &=& \bigcup_{(h,r,t) \in \mathcal{K}} \mathcal{T}(h, r) \cup \mathcal{H}(r, t)
     \enspace.

Theoretically, we would need to exclude all positive triples from this set of candidates for negative
triples, i.e., $\mathcal{N}^- = \mathcal{N} \setminus \mathcal{K}$.
In practice, however, since usually $|\mathcal{N}| \gg |\mathcal{K}|$, the likelihood of generating a
false negative is rather low.
Therefore, the additional filter step is often omitted to lower computational cost.
It should be taken into account that a corrupted triple that is \textit{not part }of the \ac{kg} can
represent a true fact.

"""

from typing import Set, Type

from class_resolver import Resolver

from .callbacks import TrainingCallback  # noqa: F401
from .lcwa import LCWATrainingLoop  # noqa: F401
from .slcwa import SLCWATrainingLoop  # noqa: F401
from .training_loop import NonFiniteLossError, TrainingLoop  # noqa: F401

__all__ = [
    'TrainingLoop',
    'SLCWATrainingLoop',
    'LCWATrainingLoop',
    'NonFiniteLossError',
    'training_loop_resolver',
    'TrainingCallback',
]

training_loop_resolver = Resolver.from_subclasses(
    TrainingLoop,
    default=SLCWATrainingLoop,
)
