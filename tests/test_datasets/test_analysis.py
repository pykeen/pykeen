# -*- coding: utf-8 -*-

"""Tests for dataset analysis utilities."""

import itertools
import unittest
from typing import Iterable, Mapping

import numpy as np
import pandas

from pykeen.datasets import Dataset, Nations
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
        pairs = list(
            zip(
                np.random.randint(low=0, high=200, size=n, dtype=int),
                np.random.uniform(0, 6, size=n),
            )
        )
        self.assertEqual(set(_old_skyline(pairs)), set(triple_analysis._get_skyline(pairs)))


def _test_count_dataframe(
    dataset: Dataset,
    df: pandas.DataFrame,
    labels: bool = True,
    merge_subsets: bool = True,
    merge_sides: bool = True,
):
    """Check the general structure of a count dataframe."""
    # check correct output type
    assert isinstance(df, pandas.DataFrame)

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
    df: pandas.DataFrame,
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
        df = dataset_analysis.get_relation_pattern_types_df(
            dataset=self.dataset,
            drop_confidence=False,
        )

        # check correct type
        assert isinstance(df, pandas.DataFrame)

        # check relation_id value range
        assert df[triple_analysis.RELATION_ID_COLUMN_NAME].isin(self.dataset.relation_to_id.values()).all()

        # check pattern value range
        assert df[triple_analysis.PATTERN_TYPE_COLUMN_NAME].isin(triple_analysis.RELATION_PATTERN_TYPES).all()

        # check confidence value range
        x = df[triple_analysis.CONFIDENCE_COLUMN_NAME].values
        assert (0 <= x).all()
        assert (x <= 1).all()

        # check support value range
        x = df[triple_analysis.SUPPORT_COLUMN_NAME].values
        assert (1 <= x).all()

    def test_relation_cardinality_types(self):
        """Tests for relation cardinality type classification."""
        df = dataset_analysis.get_relation_cardinality_types_df(
            dataset=self.dataset,
        )

        # check correct type
        assert isinstance(df, pandas.DataFrame)

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
        assert isinstance(df, pandas.DataFrame)

        assert {
            triple_analysis.RELATION_ID_COLUMN_NAME,
            triple_analysis.FUNCTIONALITY_COLUMN_NAME,
            triple_analysis.INVERSE_FUNCTIONALITY_COLUMN_NAME,
        }.issubset(df.columns)

        # check relation_id value range
        assert df[triple_analysis.RELATION_ID_COLUMN_NAME].isin(self.dataset.relation_to_id.values()).all()
