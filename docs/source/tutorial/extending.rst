How to Extend PyKEEN
====================
The following tutorial shows how to extend PyKeen.

Interaction Models
------------------
Let's assume you invented a new interaction model, e.g. this variant of DistMult

.. math::

    f(h, r, t) = <h, \sigma(r), t>

where :math:`h,r,t \in \mathbb{R}^d`, and :math:`\sigma` denotes the logistic sigmoid.

From the `documentation <https://pykeen.readthedocs.io/en/latest/reference/models.html#module-pykeen.models.base>`_
of model base classes, we can see that
`EntityRelationEmbeddingModel <https://pykeen.readthedocs.io/en/latest/api/pykeen.models.base.EntityRelationEmbeddingModel.html#pykeen.models.base.EntityRelationEmbeddingModel>`_
is a good candidate for a base class since we want to have embeddings for entities *and* relations.

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

and afterwards we can use this new model with all available losses, evaluators, training pipelines, inverse triple modelling, ...
