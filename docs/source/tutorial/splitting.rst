.. _splitting:

Splitting
=========
In the transductive setting, we require that all entities and relations encountered during evaluation are already
known at the time of training, i.e. are contained in the training dataset.
This makes the creation of evaluation splits more difficult.

One way to deal with this situation is to randomly select training triples, and then discard any entity or relation
from the remaining triples that is not part of any training triple.
You can do this in PyKEEN by creating a :class:`TriplesFactory` for the training triples, and then passing its entity
and relation to the ID mapping to create a :class:`TriplesFactory` for the evaluation triples.
This will filter out any triple with an unknown ID (and warn you about it, so you know).

PyKEEN also offers more advanced methods for creating splits that aim to preserve more of the underlying triples used
to create the split.
We have implemented several algorithms to achieve this goal, which are described below.
The coverage-based algorithm is the default.
All of them are capable of configuring the desired split ratios between the resulting triple sets.
It may not always be possible to find a suitable split with the exact ratios, so the result may be slightly different.
In this case you will see a warning about the different ratio.

Coverage
--------
This method first selects one triple for each entity for the training part, and then randomly splits the remaining
triples with an adjusted split ratio.
In some cases with very sparse graphs or too small training ratios, it may not be possible to select the initial
coverage, and the splitting method will fail.

Cleanup
-------
The triples are first split randomly without taking into account the transductive requirements.
After that, there may be entities or relations that only occur in evaluation triples.
Thus, for each set of evaluation triples, we move triples from the evaluation set to the training set until all
entities occur in at least one training triple.
This may increase the number of training triples and thus tilt the split ratio towards training.

We offer two different cleanup methods: `randomized` and `deterministic`.

Deterministic
~~~~~~~~~~~~~
The deterministic method finds all triples that contain at least one entity or relation that is not part of the
training set and moves them all into the training set.
It takes only a single iteration and is therefore very fast, but may move more triples into the training set than
necessary, resulting in smaller evaluation sets than desired.

Randomized
~~~~~~~~~~
The randomized method is iterative:
It determines all triples that have an entity or relation that is not covered by training triples, and randomly
selects one of these triples to move into the training set.
The process stops when all entities and relations are covered.
Is usually requires multiple iterations and is therefore slower than the deterministic algorithm.
However, it may find a smaller set of triples than the deterministic one and thus come closer to the desired split
ratio.


.. warning ::
    PyKEEN currently only supports the creation of transductive splits.
    One reason for this is that in the inductive setting, there are several different ways to create inductive splits and the literature uses different ones, cf. https://arxiv.org/abs/2107.04894.
    You can still use PyKEEN for inductive link prediction with existing data sets, or with new inductive data sets that you create.
    For a general discussion of inductive link prediction, see :ref:`inductive_lp`.
