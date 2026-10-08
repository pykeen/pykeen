"""Tests for node piece."""

import random
from collections.abc import Collection, Mapping, MutableMapping
from typing import Any, ClassVar

import numpy as np
import pytest
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
    kwargs: ClassVar[Mapping[str, Any]] = {
        "selections": [
            pykeen.nn.node_piece.DegreeAnchorSelection,
            pykeen.nn.node_piece.PageRankAnchorSelection,
        ],
    }


class AnchorSelectionMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.node_piece.AnchorSelection]):
    """Test for tests for anchor selection strategies."""

    base_cls = pykeen.nn.node_piece.AnchorSelection
    base_test = cases.AnchorSelectionTestCase
    skip_cls: ClassVar[Collection[type]] = {pykeen.nn.node_piece.SingleSelection}


class CSGraphAnchorSearcherTests(cases.AnchorSearcherTestCase):
    """Tests for anchor search with scipy.sparse.csgraph."""

    cls = pykeen.nn.node_piece.CSGraphAnchorSearcher


class ChunkedCSGraphAnchorSearcherTests(cases.AnchorSearcherTestCase):
    """Tests for anchor search with scipy.sparse.csgraph, using small chunks."""

    cls = pykeen.nn.node_piece.CSGraphAnchorSearcher
    kwargs: ClassVar[Mapping[str, Any]] = {"chunk_size": 2}


@pytest.mark.parametrize(
    ("num_entities", "num_edges", "num_anchors", "k", "seed"),
    [
        (50, 60, 10, 3, 0),
        # sparse graph with many unreachable anchors
        (80, 30, 12, 4, 1),
        # k larger than the number of anchors
        (30, 50, 3, 5, 2),
        # many ties
        (100, 400, 40, 8, 3),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_csgraph_chunked(num_entities: int, num_edges: int, num_anchors: int, k: int, seed: int, chunk_size: int):
    """Test that chunked anchor search agrees with unchunked search and dense shortest path distances."""
    rng = np.random.default_rng(seed=seed)
    edge_index = rng.integers(num_entities, size=(2, num_edges))
    anchors = rng.choice(num_entities, size=num_anchors, replace=False)
    cls = pykeen.nn.node_piece.CSGraphAnchorSearcher
    expected = cls(chunk_size=num_anchors)(edge_index=edge_index, anchors=anchors, k=k, num_entities=num_entities)
    result = cls(chunk_size=chunk_size)(edge_index=edge_index, anchors=anchors, k=k, num_entities=num_entities)
    np.testing.assert_array_equal(result, expected)

    # the result is padded with -1 if there are fewer than k anchors
    assert result.shape == (num_entities, k)
    assert (result[:, num_anchors:] == -1).all()
    result = result[:, :num_anchors]

    # compare against reference distances
    k = min(k, num_anchors)
    adjacency = scipy.sparse.coo_matrix(
        (np.ones(num_edges), tuple(edge_index)), shape=(num_entities, num_entities)
    ).tocsr()
    distances = scipy.sparse.csgraph.shortest_path(
        csgraph=adjacency, directed=False, unweighted=True, indices=anchors
    ).T
    exp_dist = np.sort(distances, axis=1)[:, :k]
    # unreachable anchors are padded with -1
    reachable = np.isfinite(exp_dist)
    np.testing.assert_array_equal(result >= 0, reachable)
    # the selected anchors have the k smallest distances, in ascending order
    dist = np.take_along_axis(distances, np.maximum(result, 0), axis=1)
    np.testing.assert_array_equal(dist[reachable], exp_dist[reachable])
    # ties are broken by the smaller anchor ID
    with np.errstate(invalid="ignore"):  # inf - inf for unreachable anchors
        same = (np.diff(dist, axis=1) == 0) & reachable[:, 1:]
    assert (np.diff(result, axis=1)[same] > 0).all()


@pytest.mark.parametrize("chunk_size", [1, 2, None])
def test_csgraph_isolated_trailing_entities(chunk_size: int | None):
    """Test that trailing entities without edges are included when num_entities is given."""
    # path graph 0 - 1 - 2; entities 3, 4, 5 have no edges
    edge_index = np.asarray([[0, 1], [1, 2]])
    num_entities = 6
    # entity 4 is an isolated anchor
    anchors = np.asarray([0, 4, 2])
    searcher = pykeen.nn.node_piece.CSGraphAnchorSearcher(chunk_size=chunk_size)
    result = searcher(edge_index=edge_index, anchors=anchors, k=2, num_entities=num_entities)
    expected = np.asarray(
        [
            [0, 2],
            # tie at distance 1, broken by smaller anchor ID
            [0, 2],
            [2, 0],
            [-1, -1],
            # an isolated anchor only reaches itself
            [1, -1],
            [-1, -1],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    # without num_entities, the number of entities is inferred from the edge index
    result = searcher(edge_index=edge_index, anchors=anchors[[0, 2]], k=2)
    np.testing.assert_array_equal(result, [[0, 1], [0, 1], [1, 0]])


@pytest.mark.parametrize(("num_anchors", "num_entities", "k", "seed"), [(3, 7, 2, 0)])
def test_top_k_indices(num_anchors: int, num_entities: int, k: int, seed: int) -> None:
    """Test top-k index calculation."""
    # generate test data (with fixed seed for reproducibility)
    rng = np.random.default_rng(seed=seed)
    array = rng.uniform(size=(num_anchors, num_entities))
    # get result using argpartition
    cls = pykeen.nn.node_piece.CSGraphAnchorSearcher
    ap = cls.topk_argpartition(array=array, k=k)
    # check shape
    assert ap.shape == (k, num_entities)
    # check type
    assert np.issubdtype(ap.dtype, np.integer)
    # check value range
    np.testing.assert_array_less(-1, ap)
    np.testing.assert_array_less(ap, num_anchors)
    # check equality with argsort
    np.testing.assert_array_equal(ap, cls.topk_argsort(array=array, k=k))


class ScipySparseAnchorSearcherTests(cases.AnchorSearcherTestCase):
    """Tests for anchor search with scipy.sparse."""

    cls = pykeen.nn.node_piece.ScipySparseAnchorSearcher

    def test_bfs(self):
        """Test bfs."""
        self.instance: pykeen.nn.node_piece.ScipySparseAnchorSearcher
        k = 2
        max_iter = 3
        edge_index = np.stack([np.arange(self.num_entities - 1), np.arange(1, self.num_entities)])
        adjacency = self.instance.create_adjacency(edge_index=edge_index)
        anchors = np.arange(3)
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
        k_dist = np.partition(distances, kth=k, axis=0)[:k, :].T.max(axis=1)
        # nodes which do not reach k anchors within max_iter hops keep all anchors reachable within max_iter hops
        exp_pool = (distances <= np.minimum(k_dist, max_iter)).T

        np.testing.assert_array_equal(pool >= 0, exp_pool)
        # the pool contains the number of hops
        np.testing.assert_array_equal(pool[exp_pool], distances.T[exp_pool])

    def test_select_closest(self):
        """Test that the closest anchors are selected."""
        num_entities, num_edges, num_anchors, k = 60, 70, 15, 4
        rng = np.random.default_rng(seed=0)
        edge_index = rng.integers(num_entities, size=(2, num_edges))
        anchors = rng.choice(num_entities, size=num_anchors, replace=False)
        searcher = pykeen.nn.node_piece.ScipySparseAnchorSearcher(max_iter=num_entities)
        tokens = searcher(edge_index=edge_index, anchors=anchors, k=k, num_entities=num_entities)
        adjacency = searcher.create_adjacency(edge_index=edge_index, num_entities=num_entities)
        distances = scipy.sparse.csgraph.shortest_path(
            csgraph=adjacency, directed=False, unweighted=True, indices=anchors
        ).T
        exp_dist = np.sort(distances, axis=1)[:, :k]
        reachable = np.isfinite(exp_dist)
        np.testing.assert_array_equal(tokens >= 0, reachable)
        dist = np.take_along_axis(distances, np.maximum(tokens, 0), axis=1)
        np.testing.assert_array_equal(dist[reachable], exp_dist[reachable])


@needs_packages("torch_sparse")
class SparseBFSSearcherTests(cases.AnchorSearcherTestCase):
    """Tests for anchor search with scipy.sparse."""

    cls = pykeen.nn.node_piece.SparseBFSSearcher


class PersonalizedPageRankAnchorSearcherTests(cases.AnchorSearcherTestCase):
    """Tests for anchor search via PPR."""

    cls = pykeen.nn.node_piece.PersonalizedPageRankAnchorSearcher


@pytest.mark.parametrize("batch_size", [2, 4, 5])
def test_ppr_batch_size(batch_size: int):
    """Test PPR anchor search with a number of entities which is not divisible by the batch size."""
    num_entities = 33
    edge_index = np.stack([np.arange(num_entities - 1), np.arange(1, num_entities)])
    anchors = np.arange(0, num_entities, 10)
    cls = pykeen.nn.node_piece.PersonalizedPageRankAnchorSearcher
    ppr = np.concatenate(
        list(cls(batch_size=batch_size)._iter_ppr(edge_index=edge_index, anchors=anchors, num_entities=num_entities))
    )
    exp_ppr = np.concatenate(
        list(cls(batch_size=1)._iter_ppr(edge_index=edge_index, anchors=anchors, num_entities=num_entities))
    )
    np.testing.assert_allclose(ppr, exp_ppr, atol=1.0e-03)
    tokens = cls(batch_size=batch_size)(edge_index=edge_index, anchors=anchors, k=2, num_entities=num_entities)
    assert tokens.shape == (num_entities, 2)


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

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
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
