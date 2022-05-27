Extending the Models
====================
You should first read the tutorial on bringing your own interaction module.
This tutorial is about how to wrap a custom interaction module with a model
module for general reuse and application.

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


Instead of creating a new class, you can also directly use the :class:`pykeen.models.ERModel`, e.g.

.. code-block:: python

    from pykeen.models import ERModel
    from pykeen.nn import EmbeddingSpecification
    from pykeen.losses import BCEWithLogitsLoss

    model = ERModel(
        triples_factory=...,
        loss=BCEWithLogitsLoss(),
        interaction="transformer",
        entity_representations=EmbeddingSpecification(embedding_dim=64),
        relation_representations=EmbeddingSpecification(embedding_dim=64),
    )


.. todo::

    tutorial on rolling your own more complicated model, like :class:`pykeen.nn.modules.NTNInteraction` or
    :class:`pykeen.nn.modules.TransDInteraction`.

.. todo::

    tutorial on using some of the inheriting classes of :class:`pykeen.nn.modules.Interaction` like
    :class:`pykeen.nn.modules.FunctionalInteraction` or :class:`pykeen.nn.modules.TranslationalInteraction`
