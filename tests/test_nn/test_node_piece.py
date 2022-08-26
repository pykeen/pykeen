"""Tests for node piece."""

import random
from typing import Any, MutableMapping

import numpy
import numpy.testing
import scipy.sparse.csgraph
import unittest_templates

import pykeen.nn.node_piece
from tests import cases
from tests.utils import needs_packages


class DegreeAnchorSelectionTestCase(cases.AnchorSelectionTestCase):
    """Tests for degree anchor selection."""

    cls = pykeen.nn.node_piece.DegreeAnchorSelection


class PageRankAnchorSelectionTestCase(cases.AnchorSelectionTestCase):
    """Tests for page rank anchor selection."""

    cls = pykeen.nn.node_piece.PageRankAnchorSelection


class RandomAnchorSelectionTestCase(cases.AnchorSelectionTestCase):
    """Tests for random anchor selection."""

    cls = pykeen.nn.node_piece.RandomAnchorSelection


class MixtureAnchorSelectionTestCase(cases.AnchorSelectionTestCase):
    """Tests for mixture anchor selection."""

    cls = pykeen.nn.node_piece.MixtureAnchorSelection
    kwargs = dict(
        selections=[
            pykeen.nn.node_piece.DegreeAnchorSelection,
            pykeen.nn.node_piece.PageRankAnchorSelection,
        ],
    )


class AnchorSelectionMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.node_piece.AnchorSelection]):
    """Test for tests for anchor selection strategies."""

    base_cls = pykeen.nn.node_piece.AnchorSelection
    base_test = cases.AnchorSelectionTestCase
    skip_cls = {pykeen.nn.node_piece.SingleSelection}


class CSGraphAnchorSearcherTests(cases.AnchorSearcherTestCase):
    """Tests for anchor search with scipy.sparse.csgraph."""

    cls = pykeen.nn.node_piece.CSGraphAnchorSearcher


class ScipySparseAnchorSearcherTests(cases.AnchorSearcherTestCase):
    """Tests for anchor search with scipy.sparse."""

    cls = pykeen.nn.node_piece.ScipySparseAnchorSearcher

    def test_bfs(self):
        """Test bfs."""
        self.instance: pykeen.nn.node_piece.ScipySparseAnchorSearcher
        k = 2
        max_iter = 3
        edge_index = numpy.stack([numpy.arange(self.num_entities - 1), numpy.arange(1, self.num_entities)])
        adjacency = self.instance.create_adjacency(edge_index=edge_index)
        anchors = numpy.arange(3)
        # determine pool using anchor searcher
        pool = self.instance.bfs(
            anchors=anchors,
            adjacency=adjacency,
            max_iter=max_iter,
            k=k,
        )
        # determine expected pool using shortest path distances via scipy.sparse.csgraph
        distances = scipy.sparse.csgraph.shortest_path(
            csgraph=adjacency,
            directed=False,
            return_predecessors=False,
            unweighted=True,
            indices=anchors,
        )
        k_dist = numpy.partition(distances, kth=k, axis=0)[:k, :].T.max(axis=1)
        exp_pool = ((distances <= k_dist) & (k_dist <= max_iter)).T

        numpy.testing.assert_array_equal(pool, exp_pool)


@needs_packages("torch_sparse")
class SparseBFSSearcherTests(cases.AnchorSearcherTestCase):
    """Tests for anchor search with scipy.sparse."""

    cls = pykeen.nn.node_piece.SparseBFSSearcher


class PersonalizedPageRankAnchorSearcherTests(cases.AnchorSearcherTestCase):
    """Tests for anchor search via PPR."""

    cls = pykeen.nn.node_piece.PersonalizedPageRankAnchorSearcher


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


@needs_packages("torch_sparse")
class MetisAnchorTokenizerTests(cases.TokenizerTestCase):
    """Tests for tokenization with anchor entities and metis."""

    cls = pykeen.nn.node_piece.MetisAnchorTokenizer


class PrecomputedPoolTokenizerTests(cases.TokenizerTestCase):
    """Tests for tokenization with precomputed token pools."""

    cls = pykeen.nn.node_piece.PrecomputedPoolTokenizer

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        # generate random pool
        kwargs["pool"] = {
            i: random.sample(range(2 * self.num_tokens), k=self.num_tokens) for i in range(self.factory.num_entities)
        }
        return kwargs


class TokenizerMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.node_piece.Tokenizer]):
    """Test for tests for tokenizers."""

    base_cls = pykeen.nn.node_piece.Tokenizer
    base_test = cases.TokenizerTestCase
