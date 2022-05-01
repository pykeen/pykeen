# -*- coding: utf-8 -*-

"""
NodePiece modules.

A :class:`NodePieceRepresentation` contains a collection of :class:`TokenizationRepresentation`.
A :class:`TokenizationRepresentation` is defined as :class:`Representation` module mapping token
indices to representations, also called the `vocabulary` in resemblance of token representations
known from NLP applications, and an assignment from entities to (multiple) tokens.

In order to obtain the vocabulary and assignment, multiple options are available, which often
follow a two-step approach of first selecting a vocabulary, and afterwards assigning the entities
to the set of tokens, usually using the graph structure of the KG.

One way of tokenization, is tokenization by :class:`AnchorTokenizer`, which selects some anchor
entities from the graph as vocabulary. The anchor selection process is controlled by an
:class:`AnchorSelection` instance. In order to obtain the assignment, some measure of graph
distance is used. To this end, a :class:`AnchorSearcher` instance calculates the closest
anchor entities from the vocabulary for each of the entities in the graph.

Since some tokenizations are expensive to compute, we offer a mechanism to use precomputed tokenizations via
:class:`PrecomputedPoolTokenizer`. To enable loading from different formats, a loader subclassing from
:class:`PrecomputedTokenizerLoader` can be selected accordingly. To precompute anchor-based tokenizations,
you can use the command

.. code-block:: console

    pykeen tokenize

Its usage is explained by passing the ``--help`` flag.
"""

from .anchor_search import (
    AnchorSearcher,
    CSGraphAnchorSearcher,
    PersonalizedPageRankAnchorSearcher,
    ScipySparseAnchorSearcher,
    anchor_searcher_resolver,
)
from .anchor_selection import (
    AnchorSelection,
    DegreeAnchorSelection,
    MixtureAnchorSelection,
    PageRankAnchorSelection,
    RandomAnchorSelection,
    SingleSelection,
    anchor_selection_resolver,
)
from .loader import (
    GalkinPrecomputedTokenizerLoader,
    PrecomputedTokenizerLoader,
    TorchPrecomputedTokenizerLoader,
    precomputed_tokenizer_loader_resolver,
)
from .representations import HashDiversityInfo, NodePieceRepresentation, TokenizationRepresentation
from .tokenization import AnchorTokenizer, PrecomputedPoolTokenizer, RelationTokenizer, Tokenizer, tokenizer_resolver

__all__ = [
    # Anchor Searchers
    "anchor_searcher_resolver",
    "AnchorSearcher",
    "ScipySparseAnchorSearcher",
    "CSGraphAnchorSearcher",
    "PersonalizedPageRankAnchorSearcher",
    # Anchor Selection
    "anchor_selection_resolver",
    "AnchorSelection",
    "SingleSelection",
    "DegreeAnchorSelection",
    "MixtureAnchorSelection",
    "PageRankAnchorSelection",
    "RandomAnchorSelection",
    # Tokenizers
    "tokenizer_resolver",
    "Tokenizer",
    "RelationTokenizer",
    "AnchorTokenizer",
    "PrecomputedPoolTokenizer",
    # Token Loaders
    "precomputed_tokenizer_loader_resolver",
    "PrecomputedTokenizerLoader",
    "GalkinPrecomputedTokenizerLoader",
    "TorchPrecomputedTokenizerLoader",
    # Representations
    "TokenizationRepresentation",
    "NodePieceRepresentation",
    # Data containers
    "HashDiversityInfo",
]

# TODO: use graph library, such as igraph, graph-tool, or networkit
