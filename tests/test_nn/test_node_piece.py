"""Tests for node piece."""
import unittest_templates

import pykeen.nn.node_piece
from tests import cases


class DegreeAnchorSelectionTestCase(cases.AnchorSelectionTestCase):
    """Tests for degree anchor selection."""

    cls = pykeen.nn.node_piece.DegreeAnchorSelection


class PageRankAnchorSelectionTestCase(cases.AnchorSelectionTestCase):
    """Tests for page rank anchor selection."""

    cls = pykeen.nn.node_piece.PageRankSelection


class AnchorSelectionMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.node_piece.AnchorSelection]):
    """Test for tests for anchor selection strategies."""

    base_cls = pykeen.nn.node_piece.AnchorSelection
    base_test = cases.AnchorSelectionTestCase


class CSGraphAnchorSearcherTests(cases.AnchorSearcherTestCase):
    """Tests for anchor search with scipy.sparse.csgraph."""

    cls = pykeen.nn.node_piece.CSGraphAnchorSearcher


class ScipySparseAnchorSearcherTests(cases.AnchorSearcherTestCase):
    """Tests for anchor search with scipy.sparse."""

    cls = pykeen.nn.node_piece.ScipySparseAnchorSearcher


class AnchorSearcherMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.node_piece.AnchorSearcher]):
    """Test for tests for anchor search strategies."""

    base_cls = pykeen.nn.node_piece.AnchorSearcher
    base_test = cases.AnchorSearcherTestCase


class RelationTokenizerTests(cases.TokenizerTestCase):
    """Tests for tokenization with relational context."""

    cls = pykeen.nn.node_piece.RelationTokenizer


class AnchorTokenizerTests(cases.TokenizerTestCase):
    """Tests for tokenization with anchor entities."""

    cls = pykeen.nn.node_piece.AnchorTokenizer


class TokenizerMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.node_piece.Tokenizer]):
    """Test for tests for tokenizers."""

    base_cls = pykeen.nn.node_piece.Tokenizer
    base_test = cases.TokenizerTestCase
