.. _representations:

Representations
===============

In PyKEEN, a :class:`~pykeen.nn.representation.Representation` is used to map *integer indices* to *numeric
representations*. A simple example is an :class:`~pykeen.nn.representation.Embedding`, where the mapping is a simple
lookup. However, more advanced representation modules are available, too.

This tutorial is intended to provide a comprehensive overview of possible components. Feel free to visit the pages of
the individual representations for detailed technical information.

.. contents:: Table of Contents
    :depth: 3

Base
----

The :class:`~pykeen.nn.representation.Representation` class defines a common interface for all representation modules.
Each representation defines a :attr:`~pykeen.nn.representation.Representation.max_id` attribute. We can pass any integer
index $i \in [0, \text{max_id})$ to a representation module to get a numeric representation of a fixed shape
:attr:`~pykeen.nn.representation.Representation.shape`.

.. note::

    To support efficient training and inference, all representations accept batches of indices of arbitrary shape, and
    return batches of corresponding numeric representations. The batch dimensions always precede the actual shape of the
    returned numerical representations.

Combinations & Adapters
-----------------------

PyKEEN provides a rich set of generic tools to combine and adapt representations to form new representations.

Transformed Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~pykeen.nn.representation.TransformedRepresentation` adds a (learnable) transformation to an existing
representation. It can be particularly useful when we have some fixed features for entities or relations, e.g. from a
pre-trained model, or encodings of other modalities like text or images, and want to learn a transformation on them to
make them suitable for simple interaction functions like :class:`~pykeen.nn.modules.DistMultInteraction`.

Subset Representations
~~~~~~~~~~~~~~~~~~~~~~

A :class:`~pykeen.nn.representation.SubsetRepresentation` allows to *"hide"* some indices. This can be useful e.g. if we
want to share some representations between modules, while others should be exclusive, e.g. we want to use inverse
relations for a message passing phase, but no inverses in scoring triples.

Partitioned Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~pykeen.nn.representation.PartitionRepresentation` uses multiple base representations and chooses exactly one
of them for each index based on a fixed mapping. :class:`~pykeen.nn.representation.BackfillRepresentation` implements a
frequently used special case, where we have two base representations, where one is the *"main"* representation and the
other is used as a backup whenever the first one fails to provide a representation. This is useful when we want to use
features or pre-trained embeddings whenever possible, and learn new embeddings for any entities or relations for which
we have no features.

Combined Representations
~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~pykeen.nn.representation.CombinedRepresentation` can be used when we have multiple sources of representations
and want to combine those into a single one. Use cases are multi-modal models, or
:class:`~pykeen.nn.node_piece.representation.NodePieceRepresentation`.

Embedding
---------

An :class:`~pykeen.nn.representation.Embedding` is the simplest representation, where the an index is mapped to a
numerical representation by a simple lookup in a table. Despite its simplicity, almost all publications on transductive
link prediction rely on embeddings to represent entities or relations.

Decomposition
-------------

Since knowledge graphs can contain a large number of entities, having independent trainable embeddings for each of them
can lead to an excessive number of trainable parameters. Therefore, methods have been developed that do not learn
independent representations, but rather have a set of base representations and create individual representations by
combining them.

Low-Rank Factorization
~~~~~~~~~~~~~~~~~~~~~~

A simple method to reduce the number of parameters is to use a low-rank decomposition of the embedding matrix, as
implemented in :class:`~pykeen.nn.representation.LowRankRepresentation`. Here, each representation is a linear
combination of shared base representations. Typically, the number of bases is chosen to be smaller than the dimension of
each base representation. Low-rank factorization can also be seen as a special case of
:class:`~pykeen.nn.representation.CombinedRepresentation` with a restricted (but very efficient) combination operation.

Tensor Train Factorization
~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~pykeen.nn.representation.TensorTrainRepresentation` uses a tensor factorization method, which can also be
interpreted as a hierarchical decomposition. The tensor train decomposition is also known as matrix product states.

Tokenization
~~~~~~~~~~~~

In :class:`~pykeen.nn.node_piece.representation.TokenizationRepresentation`, each index is associated with a fixed
number of tokens. The tokens have their own individual representations. They are concatenated to form the combined
representation.

The representation itself does not provide any means to obtain the token mapping, but it usually carries some domain
information. An example is NodePiece, which provides tokenization via anchor nodes, i.e. representative entities within
the graph neighborhood, or relation tokenization using the set of occurring relation types.

NodePiece
~~~~~~~~~

The :class:`~pykeen.nn.node_piece.representation.NodePieceRepresentation` contains one or more
:class:`~pykeen.nn.node_piece.representation.TokenizationRepresentation` and uses an additional aggregation. The
aggregation can be simple non-parametric, e.g. the dimension-wise mean, but it can also have trainable parameters
itself.

.. seealso::

    - https://towardsdatascience.com/nodepiece-tokenizing-knowledge-graphs-6dd2b91847aa
    - :ref:`getting_started_with_node_piece`

Embedding Bag
~~~~~~~~~~~~~

An :class:`~pykeen.nn.representation.EmbeddingBagRepresentation` represents each index by a bag of individual
embeddings. Each index can have a variable number of associated embeddings. To obtain the representation, the individual
embeddings are aggregated by a simple sum, mean, or max pooling operation.

This representation is less flexible than the tokenization representation and/or NodePiece, which allow more powerful
aggregations as well as arbitrary base representations. However, it has built-in support in PyTorch, cf.
:class:`~torch.nn.EmbeddingBag`.

Message Passing
---------------

Message passing representation modules enrich the representations of entities by aggregating the information from their
graph neighborhood.

RGCN
~~~~

The :class:`~pykeen.nn.message_passing.RGCNRepresentation` uses :class:`~pykeen.nn.message_passing.RGCNLayer` to pass
messages between entities. These layers aggregate representations of neighboring entities, which are first transformed
by a relation-specific linear transformation.

CompGCN
~~~~~~~

The :class:`~pykeen.nn.representation.SingleCompGCNRepresentation` enriches representations using
:class:`~pykeen.nn.representation.CompGCNLayer`, which instead uses a more flexible composition of entity and relation
representations along each edge. As a technical detail, since each :class:`~pykeen.nn.representation.CompGCNLayer`
transforms entity and relation representations, we must first construct a
:class:`~pykeen.nn.representation.CombinedCompGCNRepresentations` and then split its output into separate
:class:`~pykeen.nn.representation.SingleCompGCNRepresentation` for entities and relations, respectively.

PyTorch Geometric
~~~~~~~~~~~~~~~~~

Another way to utilize message passing is via the modules provided in :mod:`pykeen.nn.pyg`, which allow to use the
message passing layers from PyTorch Geometric to enrich base representations via message passing. We include the
following templates to easily create custom transformations:

    - :class:`~pykeen.nn.pyg.MessagePassingRepresentation`: Base class.
    - :class:`~pykeen.nn.pyg.SimpleMessagePassingRepresentation`: For message passing ignoring relation type
      information.
    - :class:`~pykeen.nn.pyg.TypedMessagePassingRepresentation` For message passing using categorical relation type
      information.
    - :class:`~pykeen.nn.pyg.FeaturizedMessagePassingRepresentation` For message passing using relation representations
      during message passing.

Text-based
----------

Text-based representations use the entities' (or relations') labels to derive representations. To this end,
:class:`~pykeen.nn.representation.TextRepresentation` uses a (pre-trained) transformer model from the
:mod:`transformers` library to encode the labels. Since the transformer models have been trained on huge corpora of
text, their text encodings often contain semantic information, i.e., labels with similar semantic meaning get similar
representations. While we can also benefit from these strong features by just initializing an
:class:`~pykeen.nn.representation.Embedding` with the vectors, e.g., using
:class:`~pykeen.nn.init.LabelBasedInitializer`, the :class:`~pykeen.nn.representation.TextRepresentation` include the
transformer model as part of the KGE model, and thus allow fine-tuning the language model for the KGE task. This is
beneficial, e.g., since it allows a simple form of obtaining an inductive model, which can make predictions for entities
not seen during training.

.. literalinclude:: ../examples/nn/representation/text_based.py
    :lines: 3-27

We can use the label-encoder part to generate representations for unknown entities with labels. For instance, `"uk"` is
an entity in `nations`, but we can also put in `"united kingdom"`, and get a roughly equivalent vector representations

.. literalinclude:: ../examples/nn/representation/text_based.py
    :lines: 30-33

Thus, if we would put the resulting representations into the interaction function, we would get similar scores

.. literalinclude:: ../examples/nn/representation/text_based.py
    :lines: 34-

As a downside, this will usually substantially increase the computational cost of computing triple scores.

Wikidata
~~~~~~~~

Since quite a few benchmark datasets for link prediction on knowledge graphs use `Wikidata <https://www.wikidata.org>`_
as a source, e.g., :class:`~pykeen.datasets.codex.CoDExSmall` or :class:`~pykeen.datasets.wd50k.WD50KT`, we added a
convenience class :class:`~pykeen.nn.representation.WikidataTextRepresentation` that looks up labels based on Wikidata
QIDs (e.g., `Q42 <https://www.wikidata.org/wiki/Q42>`_ for Douglas Adams).

Biomedical Entities
~~~~~~~~~~~~~~~~~~~

If your dataset is labeled with compact uniform resource identifiers (e.g., CURIEs) for biomedical entities like
chemicals, proteins, diseases, and pathways, then the :class:`~pykeen.nn.representation.BiomedicalCURIERepresentation`
representation can make use of :mod:`pyobo` to look up names (via CURIE) via the :func:`pyobo.get_name` function, then
encode them using the text encoder.

All biomedical knowledge graphs in PyKEEN (at the time of adding this representation), unfortunately do not use CURIEs
for referencing biomedical entities. In the future, we hope this will change.

To learn more about CURIEs, please take a look at the `Bioregistry <https://bioregistry.io>`_ and `this blog post on
CURIEs <https://cthoyt.com/2021/09/14/curies.html>`_.

Visual
------

Sometimes, we also have visual information about entities, e.g., in the form of images. For these cases there is
:class:`~pykeen.nn.vision.representation.VisualRepresentation` which uses an image encoder backbone to obtain
representations.

Wikidata
~~~~~~~~

As for textual representations, we provide a convenience class
:class:`~pykeen.nn.vision.representation.WikidataVisualRepresentation` for Wikidata-based datasets that looks up labels
based on Wikidata QIDs.
