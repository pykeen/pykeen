.. _interactions:

Interaction Functions
=====================

In PyKEEN, an *interaction function* refers to a function that maps *representations* for head entities, relations, and
tail entities to a scalar plausibility score. In the simplest case, head entities, relations, and tail entities are each
represented by a single tensor. However, there are also interaction functions that use multiple tensors, e.g.
:class:`~pykeen.nn.modules.NTNInteraction`.

Interaction functions can also have trainable parameters that are global and not related to a single entity or relation.
An example is :class:`~pykeen.nn.modules.TuckERInteraction` with its core tensor. We call such functions stateful and
all others stateless.

Base
----

:class:`~pykeen.nn.modules.Interaction` is the base class for all interactions. It defines the API for (broadcastable,
batch) calculation of plausibility scores, see :meth:`~pykeen.nn.modules.Interaction.forward`. It also provides some
meta information about required symbolic shapes of different arguments.

Combinations & Adapters
-----------------------

The :class:`~pykeen.nn.modules.DirectionAverageInteraction` calculates a plausibility by averaging the plausibility
scores of a base function over the forward and backward representations. It can be seen as a generalization of
:class:`~pykeen.nn.modules.SimplEInteraction`.

The :class:`~pykeen.nn.modules.MonotonicAffineTransformationInteraction` adds trainable scalar scale and bias terms to
an existing interaction function. The scale parameter is parametrized to take only positive values, preserving the
interpretation of larger values corresponding to more plausible triples. This adapter is particularly useful for base
interactions with a restricted range of values, such as norm-based interactions, and loss functions with absolute
decision thresholds, such as point-wise losses, e.g., :class:`~pykeen.losses.BCEWithLogitsLoss`.

The :class:`~pykeen.nn.modules.ClampedInteraction` constrains the scores to a given range of values. While this ensures
that scores cannot exceed the bounds, using :func:`torch.clamp()` also means that no gradients are propagated for inputs
with out-of-bounds scores. It can also lead to tied scores during evaluation, which can cause problems with some
variants of the score functions, see :ref:`understanding-evaluation`.

Norm-Based Interactions
-----------------------

Norm-based interactions can be generally written as

.. math::

    -\|g(\mathbf{h}, \mathbf{r}, \mathbf{t})\|

for some (vector) norm $\|\cdot\|$ and inner function $g$. Sometimes, the $p$-th power of a $p$ norm is used instead.

Unstructured Model (UM)
~~~~~~~~~~~~~~~~~~~~~~~

The unstructured model (UM) interaction, :class:`~pykeen.nn.modules.UMInteraction`, uses the distance between head and
tail representation $\mathbf{h}, \mathbf{t} \in \mathbb{R}^d$ as inner function:

.. math::

    \mathbf{h}  - \mathbf{t}

Structure Embedding
~~~~~~~~~~~~~~~~~~~

:class:`~pykeen.nn.modules.SEInteraction` can be seen as an extension of UM, where head and relation representation
$\mathbf{h}, \mathbf{t} \in \mathbb{R}^d$ are first linearly transformed using a relation-specific head and tail
transformation matrices $\mathbf{R}_h, \mathbf{R}_t \in \mathbb{R}^{k \times d}$

.. math::

    \mathbf{R}_{h} \mathbf{h}  - \mathbf{R}_t \mathbf{t}

TransE
~~~~~~

:class:`~pykeen.nn.modules.TransEInteraction` interprets the relation representation as translation vector and defines

.. math::

    \mathbf{h} + \mathbf{r} - \mathbf{t}

for $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$

TransR
~~~~~~

:class:`~pykeen.nn.modules.TransRInteraction` uses a relation-specific projection matrix $\mathbf{R} \in \mathbb{R}^{k
\times d}$ to project $\mathbf{h}, \mathbf{t} \in \mathbb{R}^{d}$ into the relation subspace, and then applies a
:class:`~pykeen.nn.modules.TransEInteraction`-style translation by $\mathbf{r} \in \mathbb{R}^{k}$:

.. math::

    c(\mathbf{R}\mathbf{h}) + \mathbf{r} - c(\mathbf{R}\mathbf{t})

$c$ refers to an additional norm-clamping function.

TransD
~~~~~~

:class:`~pykeen.nn.modules.TransDInteraction` extends :class:`~pykeen.nn.modules.TransRInteraction` to construct
separate head and tail projections, $\mathbf{M}_{r, h}, \mathbf{M}_{r, t} \in \mathbb{R}^{k \times d}$, similar to
:class:`~pykeen.nn.modules.SEInteraction`. These projections are build (low-rank) from a shared relation-specific part
$\mathbf{r}_p \in \mathbb{R}^{k}$, and an additional head/tail representation, $\mathbf{h}_p, \mathbf{t}_p \in
\mathbb{R}^{d}$. The matrices project the base head and tail representations $\mathbf{h}_v, \mathbf{t}_v \in
\mathbb{R}^{d}$ into a relation-specific sub-space before a translation $\mathbf{r}_v \in \mathbb{R}^{k}$ is applied.

.. math::

    c(\mathbf{M}_{r, h} \mathbf{h}_v) + \mathbf{r}_v - c(\mathbf{M}_{r, t} \mathbf{t}_v)

where

.. math::

    \mathbf{M}_{r, h} &=& \mathbf{r}_p \mathbf{h}_p^{T} + \tilde{\mathbf{I}} \\
    \mathbf{M}_{r, t} &=& \mathbf{r}_p \mathbf{t}_p^{T} + \tilde{\mathbf{I}}

$c$ refers to an additional norm-clamping function.

TransH
~~~~~~

:class:`~pykeen.nn.modules.TransHInteraction` projects head and tail representations $\mathbf{h}, \mathbf{t} \in
\mathbb{R}^{d}$ to a relation-specific hyper-plane defined by $\mathbf{r}_{w} \in \mathbf{R}^d$, before applying the
relation-specific translation $\mathbf{r}_{d} \in \mathbb{R}^d$.

.. math::

    \mathbf{h}_{r} + \mathbf{r}_d - \mathbf{t}_{r}

where

.. math::

    \mathbf{h}_{r} &=& \mathbf{h} - \mathbf{r}_{w}^T \mathbf{h} \mathbf{r}_w \\
    \mathbf{t}_{r} &=& \mathbf{t} - \mathbf{r}_{w}^T \mathbf{t} \mathbf{r}_w

PairRE
~~~~~~

:class:`~pykeen.nn.modules.PairREInteraction` modulates the head and tail representations $\mathbf{h}, \mathbf{t} \in
\mathbb{R}^{d}$ by elementwise multiplication by relation-specific $\mathbf{r}_h, \mathbf{r}_t \in \mathbb{R}^{d}$,
before taking their difference

.. math::

    \mathbf{h} \odot \mathbf{r}_h - \mathbf{t} \odot \mathbf{r}_t

LineaRE
~~~~~~~

:class:`~pykeen.nn.modules.LineaREInteraction` adds an additional relation-specific translation $\mathbf{r} \in
\mathbb{R}^d$ to :class:`~pykeen.nn.modules.PairREInteraction`.

.. math::

    \mathbf{h} \odot \mathbf{r}_h - \mathbf{t} \odot \mathbf{r}_t + \mathbf{r}

TripleRE
~~~~~~~~

:class:`~pykeen.nn.modules.TripleREInteraction` adds an additional global scalar term $u \in \mathbb{r}$ to the
modulation vectors :class:`~pykeen.nn.modules.LineaREInteraction`.

.. math::

    \mathbf{h} \odot (\mathbf{r}_h + u) - \mathbf{t} \odot (\mathbf{r}_t + u) + \mathbf{r}

RotatE
~~~~~~

:class:`~pykeen.nn.modules.RotatEInteraction` uses

.. math::

    \mathbf{h} \odot \mathbf{r} - \mathbf{t}

with complex representations $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{C}^d$. When $\mathbf{r}$ is element-wise
normalized to unit length, this operation corresponds to dimension-wise rotation in the complex plane.

.. todo::

    - :class:`~pykeen.nn.modules.BoxEInteraction`
    - has some extra projections
    - :class:`~pykeen.nn.modules.MuREInteraction`
    - has some extra head/tail biases
    - :class:`~pykeen.nn.modules.TorusEInteraction`

Semantic Matching / Factorization
---------------------------------

All *semantic matching* or *factorization-based* interactions can be expressed as

.. math::

    \sum \mathbf{Z}_{i, j, k} \mathbf{h}_i \mathbf{r}_j \mathbf{t}_k

for suitable tensor $\mathbf{Z} \in \mathbb{R}^{d_h \times d_r \times d_t}$, and potentially re-shaped head entity,
relation, and tail entity representations $\mathbf{h} \in \mathbb{R}^{d_h}, \mathbf{r} \in \mathbb{R}^{d_r}, \mathbf{t}
\in \mathbb{R}^{d_t}$. Many of the interactions have a regular structured choice for $\mathbf{Z}$ which permits
efficient calculation. We will use the simplified formulae where possible.

DistMult
~~~~~~~~

The :class:`~pykeen.nn.modules.DistMultInteraction` uses the sum of products along each dimension

.. math::

    \sum_i \mathbf{h}_i \mathbf{r}_i \mathbf{t}_i

for $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$.

Canonical Tensor Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~pykeen.nn.modules.CPInteraction` is equivalent to :class:`~pykeen.nn.modules.DistMultInteraction`, except that
it uses different sources for head and tail representations, while :class:`~pykeen.nn.modules.DistMultInteraction` uses
one shared entity embedding matrix.

.. math::

    \sum_{i, j} \mathbf{h}_{i, j} \mathbf{r}_{i, j} \mathbf{t}_{i, j}

SimplE
~~~~~~

:class:`~pykeen.nn.modules.SimplEInteraction` defines the interaction as

.. math::

    \frac{1}{2} \left(
        \langle \mathbf{h}_h, \mathbf{r}_{\rightarrow}, \mathbf{t}_t \rangle
        + \langle \mathbf{t}_h, \mathbf{r}_{\leftarrow}, \mathbf{h}_t \rangle
    \right)

for $\mathbf{h}_h, \mathbf{h}_t, \mathbf{r}_{\rightarrow}, \mathbf{r}_{\leftarrow}, \mathbf{t}_{h}, \mathbf{t}_{t} \in
\mathbb{R}^{d}$. In contrast to :class:`~pykeen.nn.modules.CPInteraction`, :class:`~pykeen.nn.modules.SimplEInteraction`
introduces separate weights for each relation $\mathbf{r}_{\rightarrow}$ and $\mathbf{r}_{\leftarrow}$ for the inverse
relation.

RESCAL
~~~~~~

:class:`~pykeen.nn.modules.RESCALInteraction` operates on $\mathbf{h}, \mathbf{t} \in \mathbb{R}^d$ and $\mathbf{R} \in
\mathbb{R}^{d \times d}$ by

.. math::

    \sum_{i, j} \mathbf{h}_{i} \mathbf{R}_{i,j} \mathbf{t}_{j}

Tucker Decomposition
~~~~~~~~~~~~~~~~~~~~

:class:`~pykeen.nn.modules.TuckERInteraction` / :class:`~pykeen.nn.modules.MultiLinearTuckerInteraction` are stateful
interaction functions which make $\mathbf{Z}$ a trainable global parameter and set $d_h = d_t$.

.. math::

    \sum \mathbf{Z}_{i, j, k} \mathbf{h}_i \mathbf{r}_j \mathbf{t}_k

.. warning::

    Both additionally add batch normalization and dropout layers, which technically makes them neural models. However,
    the intuition behind the interaction is still similar to semantic matching based models, which is why we list them
    here.

DistMA
~~~~~~

:class:`~pykeen.nn.modules.DistMAInteraction` uses the sum of pairwise scalar products between $\mathbf{h}, \mathbf{r},
\mathbf{t} \in \mathbb{R}^{d}$:

.. math::

    \langle \mathbf{h}, \mathbf{r} \rangle
    + \langle \mathbf{r}, \mathbf{t} \rangle
    + \langle \mathbf{t}, \mathbf{h} \rangle

TransF
~~~~~~

:class:`~pykeen.nn.modules.TransFInteraction` defines the interaction between $\mathbf{h}, \mathbf{r}, \mathbf{t} \in
\mathbb{R}^{d}$ as:

.. math::

    2 \cdot \langle \mathbf{h}, \mathbf{t} \rangle
    + \langle \mathbf{r}, \mathbf{t} \rangle
    - \langle \mathbf{h}, \mathbf{r} \rangle

ComplEx
~~~~~~~

:class:`~pykeen.nn.modules.ComplExInteraction` extends :class:`~pykeen.nn.modules.DistMultInteraction` to use complex
numbers instead, i.e., operate on $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbf{C}^{d}$, and defines

.. math::

    \textit{Re}\left(
        \sum_i \mathbf{h}_i \mathbf{r}_i \bar{\mathbf{t}}_i
    \right)

where *Re* refers to the real part, and $\bar{\cdot}$ denotes the complex conjugate.

QuatE
~~~~~

:class:`~pykeen.nn.modules.QuatEInteraction` uses

.. math::

    \langle
        \mathbf{h} \otimes \mathbf{r},
        \mathbf{t}
    \rangle

for quaternions $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbf{H}^{d}$, and Hamilton product $\otimes$.

HolE
~~~~

:class:`~pykeen.nn.modules.HolEInteraction` is given by

.. math::

    \langle \mathbf{r}, \mathbf{h} \star \mathbf{t}\rangle

where $\star: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}^d$ denotes the circular correlation:

.. math::

    [\mathbf{a} \star \mathbf{b}]_i = \sum_{k=0}^{d-1} \mathbf{a}_{k} * \mathbf{b}_{(i+k) \mod d}

AutoSF
~~~~~~

:class:`~pykeen.nn.modules.AutoSFInteraction` is an attempt to parametrize *block-based* semantic matching interaction
functions to enable automated search across those. Its interaction is given as

.. math::

    \sum_{(i_h, i_r, i_t, s) \in \mathcal{C}} s \cdot \langle h[i_h], r[i_r], t[i_t] \rangle

where $\mathcal{C}$ defines the block interactions, and $h, r, t$ are lists of blocks.

Neural Interactions
-------------------

All other interaction functions are usually called *neural*. They share that they usually have a multi-layer
architecture (usually two) and employ non-linearities. Many of them also introduce customized hidden layers such as
interpreting concatenated embedding vectors as image, pairs of embedding vectors as normal distributions, or semantic
matching inspired sums of linear products.

Moreover, some choose a form that can be decomposed into

.. math::

    f(\mathbf{h}, \mathbf{r}, \mathbf{t}) = f_o(f_i(\mathbf{h}, \mathbf{r}), \mathbf{t})

with an expensive $f_i$ and a cheap $f_o$. Such form allows efficient scoring of many tails for a given head-relation
combination, and can becombined with inverse relation modelling for an overall efficient training and inference
architecture.

ConvE
~~~~~

:class:`~pykeen.nn.modules.ConvEInteraction` uses an interaction of the form

.. math::

    \langle g(\mathbf{h}, \mathbf{r}), \mathbf{t} \rangle + t_b

for $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$ are the head entity, relation, and tail entity representation,
and $t_b \in \mathbb{R}$ is an entity bias. $g$ is a CNN-based encoder, which first operates on a 2D-reshaped "image"
and then flattens the output for a second linear layer. Dropout and batch normalization is utilized, too.

ConvKB
~~~~~~

:class:`~pykeen.nn.modules.ConvKBInteraction` concatenates $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$ to a $3
\times d$ "image" and applies a $3 \times 1$ convolution. The output is flattened and a linear layer predicts the score.

CrossE
~~~~~~

:class:`~pykeen.nn.modules.CrossEInteraction` uses

.. math::

    \langle g(\mathbf{h}, \mathbf{r}), \mathbf{t} \rangle

where

.. math::

    g(\mathbf{h}, \mathbf{r}) = \sigma(
        \mathbf{c}_r \odot \mathbf{h}
        + \mathbf{c}_r \odot \mathbf{h} \odot \mathbf{r}
        + \mathbf{b}
    )

with an activation function $\sigma$ and $\odot$ denoting the element-wise product. Moreover, dropout is applied to the
output of $g$.

ERMLP
~~~~~

:class:`~pykeen.nn.modules.ERMLPInteraction` uses a simple 2-layer MLP on the concatenated head, relation, and tail
representations $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$.

ERMLP (E)
~~~~~~~~~

:class:`~pykeen.nn.modules.ERMLPEInteraction` adjusts :class:`~pykeen.nn.modules.ERMLPInteraction` for a more efficient
training and inference architecture by using

.. math::

    \langle g(\mathbf{h}, \mathbf{r}), \mathbf{t} \rangle

where $g$ is a 2-layer MLP.

KG2E
~~~~

:class:`~pykeen.nn.modules.KG2EInteraction` interprets pairs of vectors $\mathbf{h}_{\mu}, \mathbf{h}_{\Sigma},
\mathbf{r}_{\mu}, \mathbf{r}_{\Sigma}, \mathbf{t}_{\mu}, \mathbf{t}_{\Sigma} \in \mathbb{R}^d$ as normal distributions
$\mathcal{N}_h, \mathcal{N}_r, \mathcal{N}_t$ and determines a similarity between $\mathcal{N}_h - \mathcal{N}_t$ and
$\mathcal{N}_r$.

.. todo:: This does not really fit well into the neural category.

Neural Tensor Network (NTN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~pykeen.nn.modules.NTNInteraction` defines the interaction function as

.. math::

    \left \langle
        \mathbf{r}_{u},
        \sigma(
            \mathbf{h} \mathbf{R}_{3} \mathbf{t}
            + \mathbf{R}_{2} [\mathbf{h};\mathbf{t}]
            + \mathbf{r}_1
        )
    \right \rangle

where $\mathbf{h}, \mathbf{t} \in \mathbf{R}^d$ are head and tail entity representations, and $\mathbf{r}_1,
\mathbf{r}_u \in \mathbb{R}^d, \mathbf{R}_2 \in \mathbb{R}^{k \times 2d}, \mathbf{R}_3 \in \mathbf{R}^{d \times d \times
k}$ are relation-specific parameters, and $\sigma$ is an activation.

ProjE
~~~~~

:class:`~pykeen.nn.modules.ProjEInteraction` uses

.. math::

    \sigma_1(
        \left \langle
        \sigma_2(
            \mathbf{d}_h \odot \mathbf{h}
            + \mathbf{d}_r \odot \mathbf{r}
            + \mathbf{b}
        ),
        \mathbf{t}
        \right \rangle
        + b_p
    )

where $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$ are the head entity, relation, and tail entity
representations, $\mathbf{d}_h, \mathbf{d}_r, \mathbf{b} \in \mathbb{R}^d$ and $b_p \in \mathbb{R}$ are global
parameters, and $\sigma_1, \sigma_2$ activation functions.

Transformer
~~~~~~~~~~~

:class:`~pykeen.nn.modules.TransformerInteraction` uses

.. math::

    \langle g([\mathbf{h}; \mathbf{r}]), \mathbf{t} \rangle

with $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$ and $g$ denoting a transformer encoder with learnable
absolute positional embedding followed by sum pooling and a linear projection.
