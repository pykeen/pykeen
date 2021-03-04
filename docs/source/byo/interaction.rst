Bring Your Own Interaction
==========================
This is a tutorial about how to implement your own interactions
(also known as scoring functions) as subclasses of
:class:`pykeen.nn.modules.Interaction` for use in PyKEEN.
Imagine you've taken a time machine back to 2013 and you have just
invented TransE [bordes2013]_, defined as:

.. math::

    f(h, r, t) = -\| \mathbf{e}_h + \mathbf{r}_r - \mathbf{e}_t \|_2^2

where $\mathbf{e}_i$ is the $d$-dimensional embedding for entity $i$,
$\mathbf{r}_j$ is the $d$-dimensional embedding for relation $j$, and
$\|...\|_2$ is the $L_2$ norm.

To implement TransE in PyKEEN, you need to subclass the
:class:`pykeen.nn.modules.Interaction`. This class it itself
a subclass of :class:`torch.nn.Module`, which means that you need to provide
an implementation of :meth:`torch.nn.Module.forward`. However, the arguments
are predefined as ``h``, ``r``, and ``t``, which correspond to the embeddings
of the head, relation, and tail, respectively.

.. code-block:: python

    from pykeen.nn.modules import Interaction

    class TransEInteraction(Interaction):
        def forward(self, h, r, t):
            return -(h + r - t).norm(p=2, dim=-1)

Note the ``dim=-1`` because this operation is actually defined over
an entire batch of head, relation, and tail embeddings.

As another example, let's try DistMult [yang2014]_, defined as:

.. math::

    f(h, r, t) = <\mathbf{e}_h, \mathbf{r}_r, \mathbf{e}_t>

where $\mathbf{e}_i$ is the $d$-dimensional embedding for entity $i$,
$\mathbf{r}_j$ is the $d$-dimensional embedding for relation $j$, and
$<x,y,z>$ is the tensor product for $x,y,z \in \mathcal{R}^d$.

.. code-block:: python

    from pykeen.nn.modules import Interaction

    class DistMultInteraction(Interaction):
        def forward(self, h, r, t):
            return (h * r * t).sum(dim=-1)

Interactions with Hyper-Parameters
----------------------------------
While we previously defined TransE with the $L_2$ norm, it could be calculated with
a different value for $p$:

.. math::

    f(h, r, t) = -\| \mathbf{e}_h + \mathbf{r}_r - \mathbf{e}_t \|_p^2

This could be incorporated into the interaction definition by using the ``__init__()``,
storing the value for $p$ in the instance, then accessing it in ``forward()``.

.. code-block:: python

    from pykeen.nn.modules import Interaction

    class TransEInteraction(Interaction):
        def __init__(self, p: int):
            super().__init__()
            self.p = p

        def forward(self, h, r, t):
            return -(h + r - t).norm(p=self.p, dim=-1)

In general, you can put whatever you want in ``__init__()`` to support the calculation of scores.

Interactions with Non-Vector Embeddings
---------------------------------------
Structured Embedding [bordes2011]_

Interactions with Multiple Embeddings
-------------------------------------
Some interactions have multiple embeddings for either the head/tail or relation, such
as PairRE, defined as:


Differences between :class:`pykeen.nn.modules.Interaction` and :class:`pykeen.models.Model`
-------------------------------------------------------------------------------------------
The high-level :func:`pipeline` function allows you to pass pre-defined subclasses
of :class:`pykeen.models.Model` such as :class:`pykeen.models.TransE` or
:class:`pykeen.models.DistMult`. These classes are high-level wrappers around the interaction
functions :class:`pykeen.nn.modules.TransEInteraction` and :class:`nn.modules.DistMultInteraction`
that are more suited for running benchmarking experiments or practical applications of knowledge
graph embeddings that include lots of information about default hyper-parameters, recommended
hyper-parameter optimization strategies, and more complex applications of regularization schemas.

As a researcher, the :class:`pykeen.nn.modules.Interaction` is a way to quickly translate
ideas into new models that can be used without all of the overhead of defining a
:class:`pykeen.models.Model`.

If you are happy with your interaction module and would like to go the next step to
making it generally reusable, check the "Extending the Models" tutorial.
