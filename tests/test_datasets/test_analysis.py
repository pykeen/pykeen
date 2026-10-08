"""Tests for dataset analysis utilities."""

import itertools
import pathlib
import tempfile
import unittest
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping
from unittest import mock

import numpy as np
import pandas as pd
import torch

from pykeen.datasets import Dataset, Kinships, Nations
from pykeen.datasets import analysis as dataset_analysis
from pykeen.triples import analysis as triple_analysis
from pykeen.typing import LABEL_HEAD, LABEL_TAIL


def _old_skyline(xs):
    # naive implementation, O(n2)
    return {(s, c) for s, c in xs if not any(s2 >= s and c2 >= c for s2, c2 in xs if (s, c) != (s2, c2))}


class TestUtils(unittest.TestCase):
    """Test skyline."""

    def test_skyline(self):
        """Test the skyline function."""
        n = 500
        rng = np.random.default_rng(seed=42)
        pairs = list(
            zip(
                rng.integers(low=0, high=200, size=n, dtype=int),
                rng.uniform(0, 6, size=n),
                strict=False,
            )
        )
        assert set(_old_skyline(pairs)) == set(triple_analysis._get_skyline(pairs))


def _test_count_dataframe(
    dataset: Dataset,
    df: pd.DataFrame,
    labels: bool = True,
    merge_subsets: bool = True,
    merge_sides: bool = True,
):
    """Check the general structure of a count dataframe."""
    # check correct output type
    assert isinstance(df, pd.DataFrame)

    expected_columns = {triple_analysis.COUNT_COLUMN_NAME}
    expected_columns.update(
        _check_labels(
            df=df,
            labels=labels,
            id_column_name=triple_analysis.ENTITY_ID_COLUMN_NAME,
            label_column_name=triple_analysis.ENTITY_LABEL_COLUMN_NAME,
            label_to_id=dataset.entity_to_id,
        )
    )
    expected_columns.update(
        _check_labels(
            df=df,
            labels=labels,
            id_column_name=triple_analysis.RELATION_ID_COLUMN_NAME,
            label_column_name=triple_analysis.RELATION_LABEL_COLUMN_NAME,
            label_to_id=dataset.relation_to_id,
        )
    )

    if not merge_subsets:
        expected_columns.add(dataset_analysis.SUBSET_COLUMN_NAME)

        # check value range subset
        assert df[dataset_analysis.SUBSET_COLUMN_NAME].isin(dataset.factory_dict.keys()).all()

    if not merge_sides:
        expected_columns.add(triple_analysis.ENTITY_POSITION_COLUMN_NAME)

        # check value range side
        assert (
            df[triple_analysis.ENTITY_POSITION_COLUMN_NAME]
            .isin(
                {
                    LABEL_HEAD,
                    LABEL_TAIL,
                }
            )
            .all()
        )

    # check columns
    assert expected_columns == set(df.columns)

    # check value range and type
    assert (df[triple_analysis.COUNT_COLUMN_NAME] >= 0).all()
    assert df[triple_analysis.COUNT_COLUMN_NAME].dtype == np.int64


def _check_labels(
    df: pd.DataFrame,
    labels: bool,
    id_column_name: str,
    label_column_name: str,
    label_to_id: Mapping[str, int],
) -> Iterable[str]:
    if id_column_name in df.columns:
        yield id_column_name

        # check value range entity IDs
        assert df[id_column_name].isin(label_to_id.values()).all()

        if labels:
            yield label_column_name

            # check value range entity labels
            assert df[label_column_name].isin(label_to_id.keys()).all()


class DatasetAnalysisTests(unittest.TestCase):
    """Tests for dataset analysis utilities."""

    def setUp(self) -> None:
        """Initialize the unittest."""
        self.dataset = Nations()

    def test_relation_count_dataframe(self):
        """Test relation count dataframe."""
        for labels, merge_subsets in itertools.product((False, True), repeat=2):
            _test_count_dataframe(
                dataset=self.dataset,
                df=dataset_analysis.get_relation_count_df(
                    dataset=self.dataset,
                    add_labels=labels,
                    merge_subsets=merge_subsets,
                ),
                labels=labels,
                merge_subsets=merge_subsets,
            )

    def test_entity_count_dataframe(self):
        """Test entity count dataframe."""
        for labels, merge_subsets, merge_sides in itertools.product((False, True), repeat=3):
            _test_count_dataframe(
                dataset=self.dataset,
                df=dataset_analysis.get_entity_count_df(
                    dataset=self.dataset,
                    add_labels=labels,
                    merge_subsets=merge_subsets,
                    merge_sides=merge_sides,
                ),
                labels=labels,
                merge_subsets=merge_subsets,
                merge_sides=merge_sides,
            )

    def test_entity_relation_co_occurrence_dataframe(self):
        """Test entity-relation co-occurrence dataframe."""
        for labels, merge_sides, merge_subsets in itertools.product((False, True), repeat=3):
            _test_count_dataframe(
                dataset=self.dataset,
                df=dataset_analysis.get_entity_relation_co_occurrence_df(
                    dataset=self.dataset,
                    merge_sides=merge_sides,
                    merge_subsets=merge_subsets,
                    add_labels=labels,
                ),
                labels=labels,
                merge_subsets=merge_subsets,
                merge_sides=merge_sides,
            )

    def test_relation_pattern_types(self):
        """Helper method for relation pattern classification."""
        # use a temporary cache directory to make sure that the patterns are actually computed (rather than loaded)
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(dataset_analysis, "PYKEEN_DATASETS", pathlib.Path(directory)),
        ):
            df = dataset_analysis.get_relation_pattern_types_df(
                dataset=self.dataset,
                drop_confidence=False,
            )
            # check that the result has been cached, and that loading it from the cache gives the same result
            assert any(pathlib.Path(directory).rglob("relation_patterns_*.tsv.xz"))
            df_cached = dataset_analysis.get_relation_pattern_types_df(
                dataset=self.dataset,
                drop_confidence=False,
            )
        pd.testing.assert_frame_equal(df.reset_index(drop=True), df_cached.reset_index(drop=True))

        # check correct type
        assert isinstance(df, pd.DataFrame)

        # check relation_id value range
        assert df[triple_analysis.RELATION_ID_COLUMN_NAME].isin(self.dataset.relation_to_id.values()).all()

        # check pattern value range
        assert df[triple_analysis.PATTERN_TYPE_COLUMN_NAME].isin(triple_analysis.RELATION_PATTERN_TYPES).all()

        # check confidence value range
        x = df[triple_analysis.CONFIDENCE_COLUMN_NAME].to_numpy()
        assert (x >= 0).all()
        assert (x <= 1).all()

        # check support value range
        x = df[triple_analysis.SUPPORT_COLUMN_NAME].to_numpy()
        assert (x >= 1).all()

    def test_relation_cardinality_types(self):
        """Tests for relation cardinality type classification."""
        df = dataset_analysis.get_relation_cardinality_types_df(
            dataset=self.dataset,
        )

        # check correct type
        assert isinstance(df, pd.DataFrame)

        # check relation_id value range
        assert df[triple_analysis.RELATION_ID_COLUMN_NAME].isin(self.dataset.relation_to_id.values()).all()

        # check pattern value range
        assert df[triple_analysis.CARDINALITY_TYPE_COLUMN_NAME].isin(triple_analysis.RELATION_CARDINALITY_TYPES).all()

    def test_calculate_relation_functionality(self):
        """Tests calculate_relation_functionality."""
        df = dataset_analysis.get_relation_functionality_df(
            dataset=self.dataset,
        )

        # check correct type
        assert isinstance(df, pd.DataFrame)

        assert {
            triple_analysis.RELATION_ID_COLUMN_NAME,
            triple_analysis.FUNCTIONALITY_COLUMN_NAME,
            triple_analysis.INVERSE_FUNCTIONALITY_COLUMN_NAME,
        }.issubset(df.columns)

        # check relation_id value range
        assert df[triple_analysis.RELATION_ID_COLUMN_NAME].isin(self.dataset.relation_to_id.values()).all()


def _old_iter_patterns(mapped_triples: Collection[tuple[int, int, int]]) -> Iterable[triple_analysis.PatternMatch]:
    """Mine patterns with the pure-Python reference implementation (pre-vectorization)."""
    pairs: dict[int, set[tuple[int, int]]] = defaultdict(set)
    adj: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    ins: dict[int, set[int]] = defaultdict(set)
    outs: dict[int, set[int]] = defaultdict(set)
    for h, r, t in mapped_triples:
        pairs[r].add((h, t))
        adj[r][h].add(t)
        outs[h].add(r)
        ins[t].add(r)
    pairs = dict(pairs)
    # unary
    for r, ht in pairs.items():
        support = len(ht)
        confidence = len(ht.intersection({(t, h) for h, t in ht})) / support
        yield triple_analysis.PatternMatch(r, triple_analysis.PATTERN_TYPE_SYMMETRY, support, confidence)
        yield triple_analysis.PatternMatch(r, triple_analysis.PATTERN_TYPE_ANTI_SYMMETRY, support, 1 - confidence)
    # binary
    for (_r1, ht1), (r, ht2) in itertools.combinations(pairs.items(), r=2):
        support = len(ht1)
        confidence = len(ht1.intersection(ht2)) / support
        yield triple_analysis.PatternMatch(r, triple_analysis.PATTERN_TYPE_INVERSION, support, confidence)
    # ternary
    candidates = {(r1, r2) for e, e_in in ins.items() if e in outs for r1, r2 in itertools.product(e_in, outs[e])}
    for r1, r2 in candidates:
        lhs = {(x, z) for x, y in pairs[r1] for z in adj[r2][y]}
        support = len(lhs)
        if not support:
            continue
        for r, ht in pairs.items():
            confidence = len(lhs.intersection(ht)) / support
            yield triple_analysis.PatternMatch(r, triple_analysis.PATTERN_TYPE_COMPOSITION, support, confidence)


def _random_triples(
    generator: np.random.Generator, num_entities: int, num_relations: int, num_triples: int
) -> list[tuple[int, int, int]]:
    """Generate random triples, including duplicates, self-loops, and reversed / copied pairs."""
    h, t = generator.integers(num_entities, size=(2, num_triples))
    r = generator.integers(num_relations, size=num_triples)
    triples = list(zip(h.tolist(), r.tolist(), t.tolist(), strict=True))
    # add some reversed triples (with random relation), copies with other relations, and duplicates
    n = num_triples // 3
    triples += [(t_, int(generator.integers(num_relations)), h_) for h_, _, t_ in triples[:n]]
    triples += [(h_, int(generator.integers(num_relations)), t_) for h_, _, t_ in triples[n : 2 * n]]
    triples += triples[:n]
    generator.shuffle(triples)
    return triples


class PatternMiningTests(unittest.TestCase):
    """Compare the vectorized relation pattern mining against a pure-Python reference implementation."""

    def _assert_same_patterns(self, mapped_triples) -> None:
        expected = Counter(_old_iter_patterns(mapped_triples))
        actual = Counter(triple_analysis.iter_patterns(mapped_triples))
        assert actual == expected
        # also check output types
        for pattern in actual:
            assert type(pattern.relation_id) is int
            assert type(pattern.support) is int
            assert type(pattern.confidence) is float

    def test_random(self):
        """Test on random graphs."""
        generator = np.random.default_rng(seed=42)
        for num_entities, num_relations, num_triples in [
            (1, 1, 1),
            (3, 2, 5),
            (5, 3, 20),
            (10, 5, 60),
            (30, 8, 300),
            (50, 20, 200),
        ]:
            with self.subTest(num_entities=num_entities, num_relations=num_relations, num_triples=num_triples):
                self._assert_same_patterns(_random_triples(generator, num_entities, num_relations, num_triples))

    def test_skip_zero(self):
        """Test skipping zero-confidence matches."""
        triples = _random_triples(np.random.default_rng(seed=3), 20, 6, 150)
        expected = Counter(p for p in triple_analysis.iter_patterns(triples) if p.confidence > 0)
        assert Counter(triple_analysis.iter_patterns(triples, skip_zero=True)) == expected

    def test_empty(self):
        """Test on empty triples."""
        assert list(triple_analysis.iter_patterns([])) == []

    def test_input_types(self):
        """Test that lists, numpy arrays, and tensors give the same result."""
        triples = _random_triples(np.random.default_rng(seed=0), 10, 4, 50)
        expected = Counter(triple_analysis.iter_patterns(triples))
        assert Counter(triple_analysis.iter_patterns(np.asarray(triples))) == expected
        assert Counter(triple_analysis.iter_patterns(torch.as_tensor(triples))) == expected

    def test_composition_chunking(self):
        """Test that the chunk size does not affect the composition statistics."""
        index = triple_analysis._build_pair_index(_random_triples(np.random.default_rng(seed=1), 20, 5, 200))
        expected = triple_analysis._composition_statistics(index)
        for chunk_size in (1, 7, 100):
            with self.subTest(chunk_size=chunk_size):
                actual = triple_analysis._composition_statistics(index, chunk_size=chunk_size)
                np.testing.assert_array_equal(actual.first, expected.first)
                np.testing.assert_array_equal(actual.second, expected.second)
                np.testing.assert_array_equal(actual.support, expected.support)
                np.testing.assert_array_equal(actual.hits.toarray(), expected.hits.toarray())

    def test_datasets(self):
        """Test on the (packaged) Nations and Kinships datasets."""
        for dataset_cls in (Nations, Kinships):
            with self.subTest(dataset=dataset_cls.__name__):
                dataset = dataset_cls()
                mapped_triples = torch.cat([f.mapped_triples for f in dataset.factory_dict.values()]).tolist()
                self._assert_same_patterns(mapped_triples)

                # check the full categorization, too
                expected = pd.DataFrame(
                    data=list(
                        triple_analysis.skyline(p for p in _old_iter_patterns(mapped_triples) if p.confidence > 0)
                    ),
                    columns=[
                        triple_analysis.RELATION_ID_COLUMN_NAME,
                        triple_analysis.PATTERN_TYPE_COLUMN_NAME,
                        triple_analysis.SUPPORT_COLUMN_NAME,
                        triple_analysis.CONFIDENCE_COLUMN_NAME,
                    ],
                )
                actual = triple_analysis.relation_pattern_types(mapped_triples=mapped_triples)
                pd.testing.assert_frame_equal(
                    actual.sort_values(by=list(actual.columns)).reset_index(drop=True),
                    expected.sort_values(by=list(expected.columns)).reset_index(drop=True),
                )
