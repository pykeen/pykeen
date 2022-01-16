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

In fact, there exist several variations of the inductive setup according to the taxonomy by [ali2021]_ :

- An inference graph is totally disconnected from the training graph (disjoint), aka *fully-inductive* setup.
  Link prediction pattern between entities is therefore *unseen-to-unseen*.
- An inference graph extends the training graph connecting new nodes to the seen graph aka *semi-inductive* setup.
  Link prediction patterns can be *unseen-to-unseen* when we predict links among newly added nodes
  or *unseen-to-seen* / *seen-to-unseen* when we predict links between known nodes and newly arrived.

PyKEEN supports inductive link prediction providing interfaces to
organize the datasets, build representations of unseen entities, and
apply any existing interaction function on top of them.
Most importantly, the set of relations **must** be seen at training time.
That is, relations seen at inference time must be a subset of training ones
because we will learn representations of those relations to transfer to unseen graphs.


Organizing the Dataset
---------------
The basic class to build inductive datasets is :class:`pykeen.datasets.inductive.InductiveDataset`.
It is supposed to contain more than 3 triple factories, i.e., in the *fully-inductive* setup it expected to have
at least 4 triple factories (`transductive_trainig`, `inductive_inference`, `inductive_validation`, `inductive_test`).
`transductive_training` is the graph with entities index `(0..N)` on which we will train a model,
`inductive_inference` is the new graph appearing at inference time with new entities (indexing `(0..K)`).
Note that the number of entities in the `transductive_training` and `inductive_inference` is different.
`inductive_validation` and `inductive_test` share the entities with `inductive_inference`
but not with `transductive_training`. This way, we inform a model that we are predicting links against the
inductive inference graph, not against the training graph.

PyKEEN supports 12 fully-inductive datasets introduced by [teru2020]_ where training and inductive inference graphs
are disjoint. Each of 3 KG families, `InductiveFB15k237`, `InductiveWN18RR`, and `InductiveNELL`, have 4 versions
varying by the size of training and inference graphs as well as the total number of entities and relations.
It is ensured that the relations sets of all inference graphs are subsets of their training graphs.


Featurizing Unseen Entities
-------------

Training entity embeddings on the training graph is meaningless as those embeddings cannot be
used at inference time.
Instead, we need some universal featurizing mechanism which would build representations of both seen
and unseen entities.
In PyKEEN, there exist at least 2 such mechanisms depending on the availability of node descriptions.


NodePiece
~~~~~~~~~
In the most basic case, unseen entities arrive without any features nor descriptions.
We cater for this case using :class:`pykeen.nn.emb.NodePieceRepresentation` -
since the set of relations at training and inference time is the same, NodePiece Representation
will *tokenize* each entity through a subset of incident relation types.
Out of computational reasons, NodePiece representations of `inductive_inference` entities
(to be seen at inference time) can be pre-computed as well.
The inductive version of NodePiece, :class:`pykeen.models.unimodel.InductiveNodePiece`, trains an encoder
on top of the vocabulary of relational *tokens* that can be easily re-used at inference time.
This way, we can obtain representations of unseen entities.
`InductiveNodePiece` can be paired with any interaction function from PyKEEN where dimension of relation vectors
is the same as dimension of final node vectors. Alternative interactions can be integrated with custom
initialization of the relation representation module.

Another example is NodePiece, which takes inspiration
from tokenization we encounter in, e.g.. NLP, and represents each entity
as a set of tokens. The implementation in PyKEEN,
:class:`pykeen.nn.emb.NodePieceRepresentation`, implements a simple yet
effective variant thereof, which uses a set of randomly chosen incident
relations (including inverse relations) as tokens.

.. seealso:: Include link to the Tutorials/Representations page


Label-based Transformer Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If entity descriptions are available, the universal featurizing mechanism can
be a language model accessible via :class:`pykeen.nn.emb.LabelBasedTransformerRepresentation`.
At both training and inference time, fixed-size entity vectors are obtained after passing
their textual descriptions through a pre-trained LM.
This is work in progress and not yet available.
TODO link to the Tutorial/Representations page


Training & Evaluation
---------------------
Generally, training and evaluation of inductive models uses similar interfaces:
sLCWA and LCWA training loops, and RankBasedEvaluator.
The important addition of inductive interfaces is the `mode` argument. When set to `mode=train`,
an inductive model has to invoke representations of the training graph, when set to `mode=valid`
or `mode=test`, the model has to invoke representations of inference graphs.
In the case of fully-inductive (disjoint) datasets from [teru2020]_ the inference graph at
validation and test is the same.

