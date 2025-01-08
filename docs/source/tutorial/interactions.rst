.. _interactions:

Interaction Functions
=====================

In PyKEEN, an *interaction function* refers to a function that maps *representations* for head entities, relations, and tail entities to a scalar plausibility score. In the simplest case, head entities, relations, and tail entities are each represented by a single tensor. However, there are also interaction functions that use multiple tensors, e.g. :class:`~pykeen.nn.modules.NTNInteraction`.

Interaction functions can also have trainable parameters that are global and not related to a single entity or relation. An example is :class:`~pykeen.nn.modules.TuckERInteraction` with its core tensor. We call such functions stateful and all others stateless.

Base
----
:class:`~pykeen.nn.modules.Interaction` is the base class for all interactions.
It defines the API for (broadcastable, batch) calculation of plausibility scores, see :meth:`~pykeen.nn.modules.Interaction.forward`.
It also provides some meta information about required symbolic shapes of different arguments.

Combinations & Adapters
-----------------------
The :class:`~pykeen.nn.modules.DirectionAverageInteraction` calculates a plausibility by averaging the plausibility scores of a base function over the forward and backward representations.
It can be seen as a generalization of :class:`~pykeen.nn.modules.SimplEInteraction`.

The :class:`~pykeen.nn.modules.MonotonicAffineTransformationInteraction` adds trainable scalar scale and bias terms to an existing interaction function. The scale parameter is parametrized to take only positive values, preserving the interpretation of larger values corresponding to more plausible triples.
This adapter is particularly useful for base interactions with a restricted range of values, such as norm-based interactions, and loss functions with absolute decision thresholds, such as point-wise losses, e.g., :class:`~pykeen.losses.BCEWithLogitsLoss`.

The :class:`~pykeen.nn.modules.ClampedInteraction` constrains the scores to a given range of values. While this ensures that scores cannot exceed the bounds, using :func:`torch.clamp()` also means that no gradients are propagated for inputs with out-of-bounds scores. It can also lead to tied scores during evaluation, which can cause problems with some variants of the score functions, see :ref:`understanding-evaluation`.


Norm-Based Interactions
-----------------------
Norm-based interactions can be generally written as

.. math ::
    -\|g(\mathbf{h}, \mathbf{r}, \mathbf{t})\|

for some (vector) norm $\|\cdot\|$ and inner function $g$.
Sometimes, the $p$-th power of a $p$ norm is used instead.

Unstructured Model (UM)
~~~~~~~~~~~~~~~~~~~~~~~
The unstructured model (UM) interaction, :class:`~pykeen.nn.modules.UMInteraction`, uses the distance between head and tail representation $\mathbf{h}, \mathbf{t} \in \mathbb{R}^d$ as inner function:

.. math ::
    \mathbf{h}  - \mathbf{t}

Structure Embedding
~~~~~~~~~~~~~~~~~~~
:class:`~pykeen.nn.modules.SEInteraction` can be seen as an extension of UM, where head and relation representation $\mathbf{h}, \mathbf{t} \in \mathbb{R}^d$ are first linearly transformed using a relation-specific head and tail transformation matrices $\mathbf{R}_h, \mathbf{R}_t \in \mathbb{R}^{k \times d}$

.. math ::

    \mathbf{R}_{h} \mathbf{h}  - \mathbf{R}_t \mathbf{t}

TransE
~~~~~~
:class:`~pykeen.nn.modules.TransEInteraction` interprets the relation representation as translation vector and defines

.. math::

    \mathbf{h} + \mathbf{r} - \mathbf{t}

for $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$

TransR
~~~~~~
:class:`~pykeen.nn.modules.TransRInteraction` uses a relation-specific projection matrix $\mathbf{R} \in \mathbb{R}^{k \times d}$ to project $\mathbf{h}, \mathbf{t} \in \mathbb{R}^{d}$ into the relation subspace, and then applies a :class:`~pykeen.nn.modules.TransEInteraction`-style translation by $\mathbf{r} \in \mathbb{R}^{k}$:

.. math ::
    c(\mathbf{R}\mathbf{h}) + \mathbf{r} - c(\mathbf{R}\mathbf{t})

$c$ refers to an additional norm-clamping function.

TransD
~~~~~~

:class:`~pykeen.nn.modules.TransDInteraction` extends :class:`~pykeen.nn.modules.TransRInteraction` to construct separate head and tail projections, $\mathbf{M}_{r, h}, \mathbf{M}_{r, t} \in \mathbb{R}^{k \times d}$ , similar to :class:`~pykeen.nn.modules.SEInteraction`.
These projections are build (low-rank) from a shared relation-specific part $\mathbf{r}_p \in \mathbb{R}^{k}$, and an additional head/tail representation, $\mathbf{h}_p, \mathbf{t}_p \in \mathbb{R}^{d}$.
The matrices project the base head and tail representations $\mathbf{h}_v, \mathbf{t}_v \in \mathbb{R}^{d}$ into a relation-specific sub-space before a translation $\mathbf{r}_v \in \mathbb{R}^{k}$ is applied.

.. math ::

    c(\mathbf{M}_{r, h} \mathbf{h}_v) + \mathbf{r}_v - c(\mathbf{M}_{r, t} \mathbf{t}_v)

where

.. math ::

    \mathbf{M}_{r, h} &=& \mathbf{r}_p \mathbf{h}_p^{T} + \tilde{\mathbf{I}} \\
    \mathbf{M}_{r, t} &=& \mathbf{r}_p \mathbf{t}_p^{T} + \tilde{\mathbf{I}}

$c$ refers to an additional norm-clamping function.

TransH
~~~~~~
:class:`~pykeen.nn.modules.TransHInteraction` projects head and tail representations $\mathbf{h}, \mathbf{t} \in \mathbb{R}^{d}$ to a relation-specific hyper-plane defined by $\mathbf{r}_{w} \in \mathbf{R}^d$, before applying the relation-specific translation $\mathbf{r}_{d} \in \mathbb{R}^d$.

.. math ::
    \mathbf{h}_{r} + \mathbf{r}_d - \mathbf{t}_{r}

where

.. math ::
    \mathbf{h}_{r} &=& \mathbf{h} - \mathbf{r}_{w}^T \mathbf{h} \mathbf{r}_w \\
    \mathbf{t}_{r} &=& \mathbf{t} - \mathbf{r}_{w}^T \mathbf{t} \mathbf{r}_w

PairRE
~~~~~~
:class:`~pykeen.nn.modules.PairREInteraction` modulates the head and tail representations $\mathbf{h}, \mathbf{t} \in \mathbb{R}^{d}$ by elementwise multiplication by relation-specific $\mathbf{r}_h, \mathbf{r}_t \in \mathbb{R}^{d}$, before taking their difference

.. math ::

    \mathbf{h} \odot \mathbf{r}_h - \mathbf{t} \odot \mathbf{r}_t

LineaRE
~~~~~~~
:class:`~pykeen.nn.modules.LineaREInteraction` adds an additional relation-specific translation $\mathbf{r} \in \mathbb{R}^d$ to :class:`~pykeen.nn.modules.PairREInteraction`.

.. math ::
    \mathbf{h} \odot \mathbf{r}_h - \mathbf{t} \odot \mathbf{r}_t + \mathbf{r}

TripleRE
~~~~~~~~
:class:`~pykeen.nn.modules.TripleREInteraction` adds an additional global scalar term $u \in \mathbb{r}$ to the modulation vectors :class:`~pykeen.nn.modules.LineaREInteraction`.

.. math ::
    \mathbf{h} \odot (\mathbf{r}_h + u) - \mathbf{t} \odot (\mathbf{r}_t + u) + \mathbf{r}

RotatE
~~~~~~
:class:`~pykeen.nn.modules.RotatEInteraction` uses

.. math ::
    \mathbf{h} \odot \mathbf{r} - \mathbf{t}

with complex representations $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{C}^d$.
When $\mathbf{r}$ is element-wise normalized to unit length, this operation corresponds to dimension-wise rotation in the complex plane.


.. todo::
    - :class:`~pykeen.nn.modules.BoxEInteraction`
        - has some extra projections
    - :class:`~pykeen.nn.modules.MuREInteraction`
        - has some extra head/tail biases
    - :class:`~pykeen.nn.modules.TorusEInteraction`
  

Semantic Matching / Factorization
----------------------------------
All *semantic matching* or *factorization-based* interactions can be expressed as

.. math ::

    \sum \mathbf{Z}_{i, j, k} \mathbf{h}_i \mathbf{r}_j \mathbf{t}_k

for suitable tensor $\mathbf{Z} \in \mathbb{R}^{d_h \times d_r \times d_t}$, and potentially re-shaped head entity, relation, and tail entity representations $\mathbf{h} \in \mathbb{R}^{d_h}, \mathbf{r} \in \mathbb{R}^{d_r}, \mathbf{t} \in \mathbb{R}^{d_t}$.
Many of the interactions have a regular structured choice for $\mathbf{Z}$ which permits efficient calculation.
We will use the simplified formulae where possible.

DistMult
~~~~~~~~
The :class:`~pykeen.nn.modules.DistMultInteraction` uses the sum of products along each dimension

.. math ::
    \sum_i \mathbf{h}_i \mathbf{r}_i \mathbf{t}_i

for $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$.

Canonical Tensor Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`~pykeen.nn.modules.CPInteraction` is equivalent to :class:`~pykeen.nn.modules.DistMultInteraction`, except that it uses different sources for head and tail representations, while :class:`~pykeen.nn.modules.DistMultInteraction` uses one shared entity embedding matrix.

.. math ::
    \sum_{i, j} \mathbf{h}_{i, j} \mathbf{r}_{i, j} \mathbf{t}_{i, j}

SimplE
~~~~~~
:class:`~pykeen.nn.modules.SimplEInteraction` defines the interaction as

.. math ::
    \frac{1}{2} \left(
        \langle \mathbf{h}_h, \mathbf{r}_{\rightarrow}, \mathbf{t}_t \rangle
        + \langle \mathbf{t}_h, \mathbf{r}_{\leftarrow}, \mathbf{h}_t \rangle
    \right)

for $\mathbf{h}_h, \mathbf{h}_t, \mathbf{r}_{\rightarrow}, \mathbf{r}_{\leftarrow}, \mathbf{t}_{h}, \mathbf{t}_{t} \in \mathbb{R}^{d}$.
In contrast to :class:`~pykeen.nn.modules.CPInteraction`, :class:`~pykeen.nn.modules.SimplEInteraction` introduces separate weights for each relation $\mathbf{r}_{\rightarrow}$ and $\mathbf{r}_{\leftarrow}$ for the inverse relation.

RESCAL
~~~~~~
:class:`~pykeen.nn.modules.RESCALInteraction` operates on $\mathbf{h}, \mathbf{t} \in \mathbb{R}^d$ and $\mathbf{R} \in \mathbb{R}^{d \times d}$ by

.. math ::
    \sum_{i, j} \mathbf{h}_{i} \mathbf{R}_{i,j} \mathbf{t}_{j}


Tucker Decomposition
~~~~~~~~~~~~~~~~~~~~
:class:`~pykeen.nn.modules.TuckERInteraction` / :class:`~pykeen.nn.modules.MultiLinearTuckerInteraction` are stateful interaction functions which make $\mathbf{Z}$ a trainable global parameter and set $d_h = d_t$.

.. math ::

    \sum \mathbf{Z}_{i, j, k} \mathbf{h}_i \mathbf{r}_j \mathbf{t}_k

.. warning::
    Both additionally add batch normalization and dropout layers, which technically makes them neural models.
    However, the intuition behind the interaction is still similar to semantic matching based models, which is why we list them here.

DistMA
~~~~~~
:class:`~pykeen.nn.modules.DistMAInteraction` uses the sum of pairwise scalar products between $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^{d}$:

.. math ::
    \langle \mathbf{h}, \mathbf{r} \rangle
    + \langle \mathbf{r}, \mathbf{t} \rangle
    + \langle \mathbf{t}, \mathbf{h} \rangle

TransF
~~~~~~
:class:`~pykeen.nn.modules.TransFInteraction` defines the interaction between $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^{d}$ as:

.. math ::
    2 \cdot \langle \mathbf{h}, \mathbf{t} \rangle
    + \langle \mathbf{r}, \mathbf{t} \rangle
    - \langle \mathbf{h}, \mathbf{r} \rangle

ComplEx
~~~~~~~
:class:`~pykeen.nn.modules.ComplExInteraction` extends :class:`~pykeen.nn.modules.DistMultInteraction` to use complex numbers instead, i.e., operate on $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbf{C}^{d}$, and defines

.. math ::
    \textit{Re}\left(
        \sum_i \mathbf{h}_i \mathbf{r}_i \bar{\mathbf{t}}_i
    \right)

where *Re* refers to the real part, and $\bar{\cdot}$ denotes the complex conjugate.

QuatE
~~~~~
:class:`~pykeen.nn.modules.QuatEInteraction` uses

.. math ::
    \langle
        \mathbf{h} \otimes \mathbf{r},
        \mathbf{t}
    \rangle

for quaternions $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbf{H}^{d}$, and Hamilton product $\otimes$.

HolE
~~~~~
:class:`~pykeen.nn.modules.HolEInteraction` is given by

.. math::
    \langle \mathbf{r}, \mathbf{h} \star \mathbf{t}\rangle

where $\star: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}^d$ denotes the circular correlation:

.. math::
    [\mathbf{a} \star \mathbf{b}]_i = \sum_{k=0}^{d-1} \mathbf{a}_{k} * \mathbf{b}_{(i+k)\ \mod \ d}

AutoSF
~~~~~~
:class:`~pykeen.nn.modules.AutoSFInteraction` is an attempt to parametrize *block-based* semantic matching interaction functions to enable automated search across those.
Its interaction is given as

.. math ::
    \sum_{(i_h, i_r, i_t, s) \in \mathcal{C}} s \cdot \langle h[i_h], r[i_r], t[i_t] \rangle

where $\mathcal{C}$ defines the block interactions, and $h, r, t$ are lists of blocks.

Neural Interactions
-------------------
All other interaction functions are usually called *neural*.

    - :class:`~pykeen.nn.modules.ConvEInteraction`
    - :class:`~pykeen.nn.modules.ConvKBInteraction`
    - :class:`~pykeen.nn.modules.CrossEInteraction`
    - :class:`~pykeen.nn.modules.ERMLPInteraction`
    - :class:`~pykeen.nn.modules.ERMLPEInteraction`
    - :class:`~pykeen.nn.modules.KG2EInteraction`
    - :class:`~pykeen.nn.modules.NTNInteraction`
    - :class:`~pykeen.nn.modules.ProjEInteraction`
    - :class:`~pykeen.nn.modules.TransformerInteraction`

Notes
=====
.. todo::
    - general description, larger is better
    - stateful vs. state-less, extra parameters
    - norm-based / semantic matching & factorization / neural
    - value ranges?
    - properties? (symmetric, etc.)
    - computational complexity?
    - expose formula programmatically?