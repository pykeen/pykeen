Extending the Interaction Models
================================
Let's assume you have invented a new interaction model,
e.g. this variant of :class:`pykeen.models.DistMult`

.. math::

    f(h, r, t) = <h, \sigma(r), t>

where :math:`h,r,t \in \mathbb{R}^d`, and :math:`\sigma` denotes the logistic sigmoid.

Picking a base class
--------------------
From the `documentation <https://pykeen.readthedocs.io/en/latest/reference/models.html#module-pykeen.models.base>`_
of model base classes, we can see that :class:`pykeen.models.base.EntityRelationEmbeddingModel`
is a good candidate for a base class since we want to have embeddings for entities *and* relations.

Implementing `score_hrt()`
--------------------------
The only implementation we have to provide is of `score_hrt`

.. code-block:: python

    from pykeen.models.base import EntityRelationEmbeddingModel

    class ModifiedDistMult(EntityRelationEmbeddingModel):
        def score_hrt(self, hrt_batch):
            # Get embeddings
            h = self.entity_embeddings(  hrt_batch[:, 0])
            r = self.relation_embeddings(hrt_batch[:, 1])
            t = self.entity_embeddings(  hrt_batch[:, 2])
            # evaluate interaction function
            return h * r.sigmoid() * t

and afterwards we can use this new model with all available losses, evaluators,
training pipelines, inverse triple modelling, via the :func:`pykeen.pipeline.pipeline`.

.. code-block:: python

    from pykeen.pipeline import pipeline

    pipeline(
        model=ModifiedDistMult,
        dataset='Nations',
        loss='NSSA',
    )

Adding defaults
---------------
You'll notice that this model is not compatible with :class:`pykeen.losses.MarginRankingLoss`
because the sigmoid causes it to be non-finite in some cases. If you have a preferred
loss function for your model, you can add the ``loss_default`` class variable
where the value is the loss class.

.. code-block:: python

    from pykeen.models.base import EntityRelationEmbeddingModel
    from pykeen.losses import NSSALoss

    class ModifiedDistMult(EntityRelationEmbeddingModel):
        loss_default = NSSALoss

        def score_hrt(self, hrt_batch):
            # Get embeddings
            h = self.entity_embeddings(  hrt_batch[:, 0])
            r = self.relation_embeddings(hrt_batch[:, 1])
            t = self.entity_embeddings(  hrt_batch[:, 2])
            # evaluate interaction function
            return h * r.sigmoid() * t

Now, when using the pipeline, the NSSA loss is used by default if none is given. The same
kind of modifications can be made to set a default regularizer with `regularizer_default`.

Implementing a custom `__init__()`
----------------------------------
Let's say you modify the previous interaction model to apply a linear transformation
to the entity embeddings using the :class:`torch.nn.Linear` module. Each PyKEEN
model is a subclass of `torch.nn.Module`, so you can update the `__init__()` function.
However, there are a couple things to consider:

1. Don't forget to properly call the `super().__init__()` and make the base class's
   arguments for `__init__()` available (even if you don't understand them). This
   is important for the pipeline to take care of automatically instantiating and
   running the code you wrote
2. Either before or after  `super().__init__()` (left to your best judgement), you
   can run any arbitrary code. Just like making normal PyTorch modules, you can
   set some submodules as attributes of the instance.
3. If your submodules need to be initialized, don't forget to implement the
   `_reset_parameters_()` function. It should call `super()._reset_parameters_()`
   function becuase there are some other nice functions already getting called.

.. math::

    f(h, r, t) = <h, \sigma(r), t>

.. code-block:: python

    import torch.nn

    from pykeen.models.base import EntityRelationEmbeddingModel
    from pykeen.losses import NSSALoss

    class ModifiedLinearDistMult(EntityRelationEmbeddingModel):
        loss_default = NSSALoss

        def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 200,
            hidden_dim: int = 20,  # extra stuff!
            automatic_memory_optimization: Optional[bool] = None,
            loss: Optional[Loss] = None,
            preferred_device: Optional[str] = None,
            random_seed: Optional[int] = None,
            regularizer: Optional[Regularizer] = None,
        ):
            super().__init__(
                triples_factory=triples_factory,
                embedding_dim=embedding_dim,
                automatic_memory_optimization=automatic_memory_optimization,
                loss=loss,
                preferred_device=preferred_device,
                random_seed=random_seed,
                regularizer=regularizer,
            )

            self.hidden_dim = hidden_dim
            self.linear = nn.Linear(self.hidden_dim, 1)

        def score_hrt(self, hrt_batch):
            # Get embeddings
            h = self.entity_embeddings(  hrt_batch[:, 0])
            h = self.linear(h)
            r = self.relation_embeddings(hrt_batch[:, 1])
            t = self.entity_embeddings(  hrt_batch[:, 2])
            t = self.linear(t)
            # evaluate interaction function
            return h * r.sigmoid() * t

        def _reset_parameters_(self):  # noqa: D102
            super()._reset_parameters_()

            # weight initialization
            nn.init.zeros_(self.linear.bias)
            nn.init.xavier_uniform_(self.linear.weight)
