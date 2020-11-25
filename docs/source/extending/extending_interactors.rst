Extending the Interaction Functions
===================================
In [ali2020]_, we argued that a knowledge graph embedding model (KGEM) consists of
several components: an interaction function, a loss function, a training approach, etc.

Let's assume you have invented a new interaction model,
e.g. this variant of :class:`pykeen.models.DistMult`

.. math::

    f(h, r, t) = <h, \sigma(r), t>

where :math:`h,r,t \in \mathbb{R}^d`, and :math:`\sigma` denotes the logistic sigmoid.

.. code-block:: python

    from pykeen.nn import Interaction

    class ModifiedDistMultInteraction(Interaction):
        def forward(self, h, r, t):
            return h * r.sigmoid() * t


.. [ali2020] Mehdi, A., *et al.* (2020) `PyKEEN 1.0: A Python Library for Training and
    Evaluating Knowledge Graph Embeddings <https://arxiv.org/abs/2007.14175>`_ *arXiv*, 2007.14175.
