r"""
A module to handle triples.

A knowledge graph can be thought of as a collection of facts, where each individual fact is represented
as a triple of a head entity $h$, a relation $r$ and a tail entity $t$. In order to operate efficiently
on them, most of PyKEEN assumes that the set of entities has been that the set of entities has been
transformed so that they are identified by integers from $[0, \ldots, E)$, where $E$ is the number of
entities. A similar assumption is made for relations with their indices of $[0, \ldots, R]$.

This module includes classes and methods for loading and transforming triples from other formats
into the index-based format, as well as advanced methods for creating leakage-free training and
test splits and analyzing data distribution.

Basic Handling
--------------
The most basic information about a knowledge graph is stored in :class:`KGInfo`. It contains the
minimal information needed to create a knowledge graph embedding: the number of entities and
relations, as well as information about the use of inverse relations (which artificially increases
the number of relations).

To store information about triples, there is the :class:`CoreTriplesFactory`. It extends
:class:`KGInfo` by additionally storing a set of index-based triples, in the form of a
3-column matrix. It also allows to store arbitrary metadata in the form of a
(JSON-compatible) dictionary. It also adds support for serialization, i.e. saving and loading
to a file, as well as filter operations and utility methods to create dataframes for
further (external) processing.

Finally, there is :class:`TriplesFactory`, which adds mapping of string-based entity and relation
names to IDs. This class also provides rich factory methods that allow creating mappings from
string-based triples alone, loading triples from sufficiently similar external file formats such
as TSV or CSV, and converting back and forth between label-based and index-based formats.
It also extends serialization to ensure that the string-to-index mappings are included along
with the files.

Splitting
---------
To evaluate knowledge graph embedding models, we need training, validation, and test sets.
In classical machine learning settings, we often have a large number of independent samples and can use simple random
sampling.
In graph learning settings, however, we are interested in learning relational patterns, i.e. patterns between different
samples.
In these settings, we need to be more careful.

For example, to evaluate models in a transductive setting, we need to make sure that all entities and relations of the
triples used in the evaluation are also present in the training triples.
PyKEEN includes methods to construct splits that ensure the presence of all entities and relations in the training part.
Those can be found in :mod:`pykeen.triples.splitting`.

In addition, knowledge graphs may contain inverse relationships, such as a *predecessor* and *successor* relationship.
In this case, careless splitting can lead to test leakage, where models that only check whether the inverse relationship
exists in training can produce significantly strong results, inflating scores without learning meaningful relationship
patterns.
PyKEEN includes methods to check knowledge graph splits for leakage, which can be found in
:mod:`pykeen.triples.leakage`.

In :mod:`pykeen.triples.remix`, we offer methods to examine the effects of a particular choice of splits.

Analysis
--------
We also provide methods for analyzing knowledge graphs.
These include simple statistics such as the number of entities or relations (in :mod:`pykeen.triples.stats`),
as well as advanced analysis of relational patterns (:mod:`pykeen.triples.analysis`).
"""

from .instances import Instances, LCWAInstances, SLCWAInstances
from .triples_factory import AnyTriples, CoreTriplesFactory, KGInfo, TriplesFactory, get_mapped_triples
from .triples_numeric_literals_factory import TriplesNumericLiteralsFactory

__all__ = [
    "Instances",
    "LCWAInstances",
    "SLCWAInstances",
    "KGInfo",
    "CoreTriplesFactory",
    "TriplesFactory",
    "TriplesNumericLiteralsFactory",
    "get_mapped_triples",
    "AnyTriples",
]
