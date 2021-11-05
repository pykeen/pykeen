Extending the Interaction Models (Old-Style)
============================================
Let's assume you have invented a new interaction model,
e.g. this variant of :class:`pykeen.models.DistMult`

.. math::

    f(h, r, t) = <h, \sigma(r), t>

where :math:`h,r,t \in \mathbb{R}^d`, and :math:`\sigma` denotes the logistic sigmoid.

Picking a Base Class
--------------------
From the reference documentation on base models (:mod:`pykeen.models.base`), we can see that
:class:`pykeen.models.base.EntityRelationEmbeddingModel` is a good candidate for a base class
since we want to have embeddings for entities *and* relations.

Implementing ``score_hrt()``
----------------------------
The only implementation we have to provide is of the `score_hrt` member function:

.. code-block:: python

    from pykeen.models.base import EntityRelationEmbeddingModel

    class ModifiedDistMult(EntityRelationEmbeddingModel):
        def score_hrt(self, hrt_batch):
            # Get embeddings
            h = self.entity_representations[0](hrt_batch[:, 0])
            r = self.relation_representations[0](hrt_batch[:, 1])
            t = self.entity_representations[0](hrt_batch[:, 2])
            # evaluate interaction function
            return h * r.sigmoid() * t

The ``entity_representations`` and ``relation_representations`` sequences are available for all
:class:`pykeen.models.base.EntityRelationEmbeddingModel` and are lists of length one containing
a single instances of a :class:`pykeen.nn.Embedding`. This may seem like a strange data structure, but
it prepares for the much more powerful usages covered by the new-style :class:`pykeen.models.ERModel`.

The ``hrt_batch`` is a long tensor representing the internal indices of the edges.
The above example shows a very common way of slicing it to get separate lists of
head indices (``hrt_batch[:, 0]``), relation indices (``hrt_batch[:, 1]``), and
tail indices (``hrt_batch[:, 2]``). Then, they are passed to the embeddings to
look up the actual values. This is vectorized, so the results are also 2-tensors
(tensors of embeddings) on which vectorized math can be applied.

Using a Custom Model with the Pipeline
--------------------------------------
We can use this new model with all available losses, evaluators,
training pipelines, inverse triple modeling, via the :func:`pykeen.pipeline.pipeline`,
since in addition to the names of models (given as strings), it can also take model
classes in the ``model`` argument.

.. code-block:: python

    from pykeen.pipeline import pipeline

    pipeline(
        model=ModifiedDistMult,
        dataset='Nations',
        loss='NSSA',
    )

Adding Defaults
---------------
If you have a preferred loss function for your model, you can add the ``loss_default`` class variable
where the value is the loss class.

.. code-block:: python

    from pykeen.models.base import EntityRelationEmbeddingModel
    from pykeen.losses import NSSALoss

    class ModifiedDistMult(EntityRelationEmbeddingModel):
        loss_default = NSSALoss

        def score_hrt(self, hrt_batch):
            h = self.entity_representations[0](hrt_batch[:, 0])
            r = self.relation_representations[0](hrt_batch[:, 1])
            t = self.entity_representations[0](hrt_batch[:, 2])
            return h * r.sigmoid() * t

Now, when using the pipeline, the :class:`pykeen.losses.NSSALoss`. loss is used by default
if none is given. The same kind of modifications can be made to set a default regularizer
with ``regularizer_default``.

Implementing a Custom `__init__()`
----------------------------------
Let's say you modify the previous interaction model to apply a two consecutive
linear transformations ``a`` and ``b`` to the entity embeddings using the :class:`torch.nn.Linear`
module.

.. math::

    f(h, r, t) = <abh, \sigma(r), abt>

Each PyKEEN model is a subclass of :class:`torch.nn.Module`, so you
can update the ``__init__()`` function. However, there are a couple things to
consider:

1. Don't forget to properly call the ``super().__init__()`` and make the base class's
   arguments for ``__init__()`` available (even if you don't understand them). This
   is important for the pipeline to take care of automatically instantiating and
   running the code you wrote
2. Either before or after  ``super().__init__()`` (left to your best judgement), you
   can run any arbitrary code. Just like making normal :mod:`torch` modules, you can
   set some submodules as attributes of the instance.
3. If your submodules need to be initialized, don't forget to implement the
   ``_reset_parameters_()`` function. It should call ``super()._reset_parameters_()``
   function because there are some parameters that could already reset by the base
   model you have chosen. This function is magically called in a post-init hook, so
   don't worry that you don't call it yourself.

.. code-block:: python

    from typing import Optional

    import torch.nn

    from pykeen.losses import Loss, NSSALoss
    from pykeen.models.base import EntityRelationEmbeddingModel
    from pykeen.pipeline import pipeline
    from pykeen.regularizers import Regularizer
    from pykeen.triples import TriplesFactory

    class ModifiedLinearDistMult(EntityRelationEmbeddingModel):
        loss_default = NSSALoss

        def __init__(
            self,
            hidden_dim: int = 20,  # extra stuff!
            **kwargs,  # pass everything else, you neither have to understand nor be able to handle the truth
        ):
            super().__init__(**kwargs)

            # Save some extra state information
            self.hidden_dim = hidden_dim

            # Note that the ``embedding_dim`` is available to all EntityRelationEmbeddingModels after init.
            self.linear1 = torch.nn.Linear(self.embedding_dim, self.hidden_dim)
            self.linear2 = torch.nn.Linear(self.hidden_dim, self.embedding_dim)

        def score_hrt(self, hrt_batch):
            h = self.entity_representations[0](hrt_batch[:, 0])
            r = self.relation_representations[0](hrt_batch[:, 1])
            t = self.entity_representations[0](hrt_batch[:, 2])

            # add more transformations
            h = self.linear2(self.linear1(h))
            t = self.linear2(self.linear1(t))

            return h * r.sigmoid() * t

        def _reset_parameters_(self):  # noqa: D102
            super()._reset_parameters_()

            # weight initialization
            torch.nn.init.zeros_(self.linear1.bias)
            torch.nn.init.zeros_(self.linear2.bias)
            torch.nn.init.xavier_uniform_(self.linear1.weight)
            torch.nn.init.xavier_uniform_(self.linear2.weight)

Adding Custom HPO Default Ranges
--------------------------------
All subclasses of :class:`pykeen.models.base.Model` can specify the default
ranges or values used during hyper-parameter optimization (HPO). PyKEEN
implements a simple dictionary-based configuration that is interpreted
by :func:`pykeen.hpo.hpo.suggest_kwargs` in the HPO pipeline.

HPO default ranges can be applied to all keyword arguments appearing in the
``__init__()`` function of your model by setting a class-level variable called
``hpo_default``.

For example, the ``hidden_dim`` can be specified as being on a range between
15 and 50 with the following:

.. code-block:: python

    class ModifiedLinearDistMult(EntityRelationEmbeddingModel):
        hpo_default = {
            'hidden_dim': dict(type=int, low=15, high=50)
        }
        ...

A step size can be imposed with ``q``:

.. code-block:: python

    class ModifiedLinearDistMult(EntityRelationEmbeddingModel):
        hpo_default = {
            'hidden_dim': dict(type=int, low=15, high=50, q=5)
        }
        ...

An alternative scale can be imposed with ``scale``. Right now, the
default is linear, and ``scale`` can optionally be set to ``power_two``
for integers as in:

.. code-block:: python

    class ModifiedLinearDistMult(EntityRelationEmbeddingModel):
        hpo_default = {
            # will uniformly give 2, 4, 8 (left inclusive, right exclusive)
            'hidden_dim': dict(type=int, low=2, high=4, scale='power_two')
        }
        ...

.. warning:: Alternative scales can not currently be used in combination with step size (``q``).

There are other possibilities for specifying the ``type`` as ``float``, ``categorical``,
or as ``bool``.

With ``float``, you can't use the ``q`` option nor set the scale to ``power_two``,
but the scale can be set to ``log`` (see :class:`optuna.distributions.LogUniformDistribution`).

.. code-block:: python

    hpo_default = {
        # will uniformly give floats on the range of [1.0, 2.0) (exclusive)
        'alpha': dict(type='float', low=1.0, high=2.0),

        # will uniformly give 1.0, 2.0, or 4.0 (exclusive)
        'beta': dict(type='float', low=1.0, high=8.0, scale='log'),
    }

With ``categorical``, you can form a dictionary like the following using ``type='categorical'``
and giving a ``choices`` entry that contains a sequence of either integers, floats, or strings.

.. code-block:: python

    hpo_default = {
        'similarity': dict(type='categorical', choices=[...])
    }

With ``bool``, you can simply use ``dict(type=bool)`` or ``dict(type='bool')``.

.. note::

    The HPO rules are subject to change as they are tightly coupled to :mod:`optuna`,
    which since version 2.0.0 has introduced several new possibilities.
