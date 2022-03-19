# -*- coding: utf-8 -*-

"""NodePiece modules."""

from .anchor_search import AnchorSearcher, CSGraphAnchorSearcher, ScipySparseAnchorSearcher, anchor_searcher_resolver
from .anchor_selection import (
    AnchorSelection,
    DegreeAnchorSelection,
    MixtureAnchorSelection,
    PageRankAnchorSelection,
    RandomAnchorSelection,
    SingleSelection,
    anchor_selection_resolver,
)
from .loader import GalkinPrecomputedTokenizerLoader, PrecomputedTokenizerLoader, precomputed_tokenizer_loader_resolver
from .representations import NodePieceRepresentation, TokenizationRepresentation
from .tokenization import AnchorTokenizer, PrecomputedPoolTokenizer, RelationTokenizer, Tokenizer, tokenizer_resolver

__all__ = [
    # Anchor Searchers
    "anchor_searcher_resolver",
    "AnchorSearcher",
    "ScipySparseAnchorSearcher",
    "CSGraphAnchorSearcher",
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
    # Representations
    "TokenizationRepresentation",
    "NodePieceRepresentation",
]

# TODO: use graph library, such as igraph, graph-tool, or networkit
