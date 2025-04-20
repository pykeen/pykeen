.. _inductive_lp:

Inductive Link Prediction
=========================
.. image:: ../img/ilp_1.png
  :alt: Transductive vs Inductive setup

For years, a standard training setup in PyKEEN and other KGE libraries
was implying that a training graph includes all entities on which we will
run inference (validation, test, or custom predictions). That is, the
missing links to be predicted connect already *seen* entities within the
train graph. Such a link prediction setup is called **transductive** setup.

What if at inference time we have new, *unseen* entities, and want to
predict links between unseen entities?
Such setups are unified under the **inductive** framework.
Illustrating the difference on the Figure above, the main difference of
the inductive setup is that at inference time we have a new graph
(called *inductive inference* graph), and link prediction is executed
against that new inference graph of unseen entities.

In fact, there exist several variations of the inductive setup according
to the taxonomy by [ali2021]_ :

- An inference graph is totally disconnected from the training graph (disjoint),
  aka *fully-inductive* setup.Link prediction pattern between entities is therefore
  *unseen-to-unseen*.
- An inference graph extends the training graph connecting new nodes to the seen
  graph aka *semi-inductive* setup. Link prediction patterns can be *unseen-to-unseen*
  when we predict links among newly added nodes or *unseen-to-seen* / *seen-to-unseen*
  when we predict links between known nodes and newly arrived.

PyKEEN supports inductive link prediction providing interfaces to
organize the datasets, build representations of unseen entities, and
apply any existing interaction function on top of them.
Most importantly, the set of relations **must** be seen at training time.
That is, relations seen at inference time must be a subset of training ones
because we will learn representations of those relations to transfer to unseen graphs.

Organizing the Dataset
----------------------
The basic class to build inductive datasets is :class:`~pykeen.datasets.inductive.InductiveDataset`.
It is supposed to contain more than 3 triple factories, i.e., in the *fully-inductive* setup it is
expected to have at least 4 triple factories (`transductive_training`, `inductive_inference`,
`inductive_validation`, `inductive_test`). `transductive_training` is the graph with entities index
`(0..N)` on which we will train a model, `inductive_inference` is the new graph appearing at inference
time with new entities (indexing `(0..K)`). Note that the number of entities in the `transductive_training`
and `inductive_inference` is different. `inductive_validation` and `inductive_test` share the entities
with `inductive_inference` but not with `transductive_training`. This way, we inform a model that we
are predicting links against the inductive inference graph, not against the training graph.

PyKEEN supports 12 fully-inductive datasets introduced by [teru2020]_ where training and inductive
inference graphs are disjoint. Each of 3 KG families, :class:`~pykeen.datasets.inductive.InductiveFB15k237`,
:class:`~pykeen.datasets.inductive.InductiveWN18RR`, and :class:`~pykeen.datasets.inductive.InductiveNELL`,
have 4 versions varying by the size of training and inference graphs as well as the total number of entities
and relations. It is ensured that the relations sets of all inference graphs are subsets of their training
graphs.


Featurizing Unseen Entities
---------------------------
Training entity embeddings on the training graph is meaningless as those embeddings cannot be
used at inference time. Instead, we need some universal featurizing mechanism which would build
representations of both seen and unseen entities. We can distinguish between cases where we have
additional features available, such as entity labels or descriptions, chemical fingerprints, ...
and cases where we only have access to relational features, i.e. the multi-relational graph structure.

.. seealso::

  The :ref:`representations` tutorial gives a general overview about different representation components.
  Some of those can be used to form inductive representations.

The following are examples of suitable representations for inductive link prediction:

Features
~~~~~~~~
In cases where you have access to features and can get them for unseen entities at inference time,
you can use a non-trainable / frozen :class:`~pykeen.nn.representation.Embedding` to store those.
This featurization method generally is domain-specific.

A special case of featurization are text-based features, e.g., from entity names or descriptions.
You can use :class:`~pykeen.nn.representation.TextRepresentation` to utilize a (usually pre-trained
text embedding or language model to obtain features. At both training and inference time, fixed-size
entity vectors are obtained after passing their textual descriptions through a pre-trained language
model.

NodePiece
~~~~~~~~~
The :class:`~pykeen.nn.NodePieceRepresentation` can be used in the most basic cases where unseen
entities arrive without any feature or descriptions. Since the set of relations at training and
inference time is the same, it will *tokenize* each entity through a subset of incident relation types.

Combination
~~~~~~~~~~~
Once you have basic inductive representations, you can also combine or transform them to form more
complex features. For example, you could add message passing or a learnable transformation to the
features.


Composing the Model
-------------------
In order to compose a link prediction model for inductive link prediction, PyKEEN offers the
:class:`~pykeen.models.InductiveERModel` base class. In comprises a mapping from *inductive modes*
to the respective entity representations. For computational reasons, these are pre-defined for
inference / validation / test and not constructed on-the-fly. However, the model exposes a utility
method to enable exchanging representations for a new inference graph, cf.
:meth:`~pykeen.models.InductiveERModel.replace_entity_representations_`.

At the moment, PyKEEN also provides two fully-configured inductive NodePiece implementations:
:class:`~pykeen.models.InductiveNodePiece` and :class:`~pykeen.models.InductiveNodePieceGNN`.
Both inductive versions of NodePiece train an encoder on top of the vocabulary of relational
*tokens* that can be easily re-used at inference time. :class:`~pykeen.models.InductiveNodePieceGNN`
additionally performs message passing over the *inductive_inference* graph after building node
representations from the vocabulary. By default, message passing is performed with a 2-layer CompGCN.
:class:`~pykeen.models.InductiveNodePiece` and :class:`~pykeen.models.InductiveNodePieceGNN` can be
paired with any interaction function from PyKEEN where the dimension of relation vectors is the same
as dimension of final node vectors. Alternative interactions can be integrated with custom
initialization of the relation representation module.

Example
~~~~~~~
Let's create a basic `InductiveNodePiece` using one of the `InductiveFB15k237` datasets:

.. literalinclude:: ../examples/tutorial/inductive_lp/01_model.py

Creating a message-passing version of NodePiece is pretty much the same:

.. literalinclude:: ../examples/tutorial/inductive_lp/01_model_gnn.py

Note this version has the ``gnn_encoder`` argument - keeping it ``None`` would invoke a default
2-layer CompGCN. You can pass here any relational GNN that returns updated matrices of entities
and relations as the scoring function will use them for ranking triples. See
:class:`~pykeen.models.InductiveNodePieceGNN` for more details.


Training & Evaluation
---------------------
Generally, training and evaluation of inductive models uses similar interfaces:
sLCWA and LCWA training loops, and RankBasedEvaluator.
The important addition of inductive interfaces is the `mode` argument. When set to `mode="training"`,
an inductive model has to invoke representations of the training graph, when set to `mode=validation`
or `mode=testing`, the model has to invoke representations of inference graphs.
In the case of fully-inductive (disjoint) datasets from [teru2020]_ the inference graph at
validation and test is the same.

By default, you can use standard PyKEEN training loops :class:`~pykeen.training.SLCWATrainingLoop` and
:class:`~pykeen.training.LCWATrainingLoop` with the new `mode` parameter. Similarly, you can use a
standard evaluator :class:`~pykeen.evaluation.rank_based_evaluator.RankBasedEvaluator` with the `mode`
parameter to evaluate validation / test triples over the whole inference graph.

Moreover, original work of [teru2020]_ used a restricted evaluation protocol ranking each
validation / test triple only against 50 random negatives. PyKEEN implements this protocol with
:class:`~pykeen.evaluation.rank_based_evaluator.SampledRankBasedEvaluator`

Let's create a training loop and validation / test evaluators:

.. literalinclude:: ../examples/tutorial/inductive_lp/02_training.py


Full Inductive LP Example
-------------------------

A minimally working example for training an `InductiveNodePieceGNN` on the `InductiveFB15k237` (v1)
in the sLCWA mode with 32 negative samples per positive, with NSSALoss, and SampledEvaluator would
look like this:

.. literalinclude:: ../examples/tutorial/inductive_lp/03_full.py
