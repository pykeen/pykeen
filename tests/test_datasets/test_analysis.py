# -*- coding: utf-8 -*-

"""Tests for dataset analysis utilities."""

import itertools
import unittest
from typing import Collection, Optional, Sequence

import numpy as np
import pandas

from pykeen.datasets import Nations
from pykeen.datasets.analysis import (SUBSET_LABELS, get_relation_functionality_df, get_entity_count_df, get_entity_relation_co_occurrence_df, get_relation_cardinality_types_df, get_relation_count_df, get_relation_pattern_types_df)
from pykeen.triples.analysis import RELATION_CARDINALITY_TYPES, RELATION_PATTERN_TYPES, _get_skyline


def _old_skyline(xs):
    # naive implementation, O(n2)
    return {
        (s, c)
        for s, c in xs
        if not any(
            s2 >= s and c2 >= c
            for s2, c2 in xs
            if (s, c) != (s2, c2)
        )
    }


class TestUtils(unittest.TestCase):
    """Test skyline."""

    def test_skyline(self):
        """Test the skyline function."""
        n = 500
        pairs = list(zip(
            np.random.randint(low=0, high=200, size=n, dtype=int),
            np.random.uniform(0, 6, size=n),
        ))
        self.assertEqual(set(_old_skyline(pairs)), set(_get_skyline(pairs)))


class AnalysisTests(unittest.TestCase):
    """Tests for dataset analysis utilities."""

    def setUp(self) -> None:
        """Initialize the unittest."""
        self.dataset = Nations()

    def _test_count_dataframe(
        self,
        df: pandas.DataFrame,
        label_name: str,
        expected_labels: Collection[str],
        first_level_column_labels: Sequence[str] = SUBSET_LABELS,
        second_level_column_labels: Optional[Sequence[str]] = None,
    ):
        """Check the general structure of a count dataframe."""
        # check correct output type
        assert isinstance(df, pandas.DataFrame)

        # check correct column (and column order)
        if second_level_column_labels is None:
            expected_columns = list(first_level_column_labels)
        else:
            expected_columns = list(itertools.product(first_level_column_labels, second_level_column_labels))
        self.assertListEqual(list(df.columns), expected_columns)

        # check that there is information for each relation
        self.assertSetEqual(set(df.index), set(expected_labels))

        # check for index name
        assert df.index.name == label_name

    def _test_count_dataframe_new(
        self,
        df: pandas.DataFrame,
        prefix: str,
    ):
        """Check the general structure of a count dataframe."""
        # check correct output type
        assert isinstance(df, pandas.DataFrame)

        # check columns
        assert {f"{prefix}_id", f"{prefix}_label", "subset", "count"}.issubset(df.columns)

        # check value range and type
        assert (df["count"] >= 0).all()
        assert df["count"].dtype == np.int64

        # check value range subset
        # TODO: Update when subset labels is fixed
        assert set(SUBSET_LABELS).union({None}).issuperset(df["subset"].unique())

    def test_relation_count_dataframe(self):
        """Test relation_count_dataframe()."""
        df = get_relation_count_df(dataset=self.dataset)
        self._test_count_dataframe_new(
            df=df,
            prefix="relation",
        )

    def test_entity_count_dataframe(self):
        """Test entity_count_dataframe()."""
        df = get_entity_count_df(dataset=self.dataset)
        self._test_count_dataframe_new(
            df=df,
            prefix="entity",
        )

    def test_entity_relation_co_occurrence_dataframe(self):
        """Test entity_relation_co_occurrence_dataframe()."""
        df = get_entity_relation_co_occurrence_df(dataset=self.dataset)

        # check correct type
        assert isinstance(df, pandas.DataFrame)

        # check index
        self.assertListEqual(
            list(df.index),
            list(itertools.product(SUBSET_LABELS, sorted(self.dataset.entity_to_id.keys()))),
        )

    def test_relation_pattern_types(self):
        """Helper method for relation pattern classification."""
        df = get_relation_pattern_types_df(
            dataset=self.dataset,
            drop_confidence=False,
        )

        # check correct type
        assert isinstance(df, pandas.DataFrame)

        # check relation_id value range
        assert df["relation_id"].isin(self.dataset.relation_to_id.values()).all()

        # check pattern value range
        assert df["pattern"].isin(RELATION_PATTERN_TYPES).all()

        # check confidence value range
        x = df["confidence"].values
        assert (0 <= x).all()
        assert (x <= 1).all()

        # check support value range
        x = df["support"].values
        assert (1 <= x).all()

    def test_relation_cardinality_types(self):
        """Tests for relation cardinality type classification."""
        df = get_relation_cardinality_types_df(
            dataset=self.dataset,
        )

        # check correct type
        assert isinstance(df, pandas.DataFrame)

        # check relation_id value range
        assert df["relation_id"].isin(self.dataset.relation_to_id.values()).all()

        # check pattern value range
        assert df["relation_type"].isin(RELATION_CARDINALITY_TYPES).all()

    def test_calculate_relation_functionality(self):
        """Tests calculate_relation_functionality."""
        df = get_relation_functionality_df(
            dataset=self.dataset,
        )

        # check correct type
        assert isinstance(df, pandas.DataFrame)

        assert {
            "relation_id",
            "functionality",
            "inverse_functionality",
        }.issubset(df.columns)

        # check relation_id value range
        assert df["relation_id"].isin(self.dataset.relation_to_id.values()).all()
