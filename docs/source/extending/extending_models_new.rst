Extending the Interaction Models
================================
In [ali2020b]_, we argued that a knowledge graph embedding model (KGEM) consists of
several components: an interaction function, a loss function, a training approach, etc.

Let's assume you have invented a new interaction model,
e.g. :class:`pykeen.models.DistMult`

.. math::

    f(h, r, t) = <h, r, t>

where :math:`h,r,t \in \mathbb{R}^d` and $<h,r,t>$ is the tensor product.

.. [ali2020b] Ali, M., *et al.* (2020) `PyKEEN 1.0: A Python Library for Training and
   Evaluating Knowledge Graph Embeddings <https://arxiv.org/abs/2007.14175>`_ *arXiv*, 2007.14175.

Implement a simple :class:`pykeen.nn.modules.Interaction`
---------------------------------------------------------
The interesting research in KGEMs is in the definition of the interaction function.
Interaction functions take a batch of embeddings for the head entities, relations, and tail
entities, and return the scores. Luckily, most operations are vectorized which means that
new interactions can be written idiomatically with PyKEEN.

.. code-block:: python

    from pykeen.nn.modules import Interaction

    class DistMultInteraction(Interaction):
        def forward(self, h, r, t):
            return h * r.sigmoid() * t

The actual implementation of DistMult's interaction function can be found at
:class:`pykeen.nn.modules.DistMultInteraction`.

Implement a simple :class:`pykeen.models.ERModel`
-------------------------------------------------
The following code block demonstrates how an interaction model can be used to define a full
KGEM using the :class:`pykeen.models.ERModel` base class.



.. code-block:: python

    from pykeen.models import ERModel
    from pykeen.nn import EmbeddingSpecification
    from pykeen.nn.modules import DistMultInteraction  # effectively the same as the example above

    class DistMult(ERModel):
        def __init__(
            self,
            # When defining your class, any hyper-parameters that can be configured should be
            # made as arguments to the __init__() function. When running the pipeline(), these
            # are passed via the ``model_kwargs``.
            embedding_dim: int = 50,
            # All remaining arguments are simply passed through to the parent constructor. If you
            # want access to them, you can name them explicitly. See the pykeen.models.ERModel
            # documentation for a full list
            **kwargs,
        ) -> None:
            # since this is a python class, you can feel free to get creative here. One example of
            # pre-processing is to derive the shape for the relation representation based on the
            # embedding dimension.
            super().__init__(
                # Pass an instance of your interaction function. This is also a place where you can
                # pass hyper-parameters, such as the L_p norm, from the KGEM to the interaction function
                interaction=DistMultInteraction(),
                # Define the entity representations using the EmbeddingSpecification. By default, each
                # embedding is linear. You can use the ``shape`` kwarg to specify higher dimensional
                # tensor shapes.
                entity_representations=EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                ),
                # Define the relation representations the same as the entities
                relation_representations=EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                ),
                # All other arguments are passed through, such as the ``triples_factory``, ``loss``,
                # ``preferred_device``, and others. These are all handled by the pipeline() function
                **kwargs,
            )

The actual implementation of DistMult can be found in :class:`pykeen.models.DistMult`. Note that
it additionally contains configuration for the initializers, constrainers, and regularizers
for each of the embeddings as well as class-level defaults for hyper-parameters and hyper-parameter
optimization. Modifying these is covered in other tutorials.

.. todo::

    tutorial on rolling your own more complicated model, like :class:`pykeen.nn.modules.NTNInteraction` or
    :class:`pykeen.nn.modules.TransDInteraction`.

.. todo::

    tutorial on using some of the inheriting classes of :class:`pykeen.nn.modules.Interaction` like
    :class:`pykeen.nn.modules.FunctionalInteraction` or :class:`pykeen.nn.modules.TranslationalInteraction`
