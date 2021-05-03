# -*- coding: utf-8 -*-

"""Tests for dataset analysis utilities."""

import itertools
import unittest

import numpy as np
import pandas

from pykeen.datasets import Nations
from pykeen.datasets.analysis import (
    SUBSET_COLUMN_NAME, get_entity_count_df, get_entity_relation_co_occurrence_df, get_relation_cardinality_types_df, get_relation_count_df,
    get_relation_functionality_df, get_relation_pattern_types_df,
)
from pykeen.triples.analysis import COUNT_COLUMN_NAME, RELATION_CARDINALITY_TYPES, RELATION_ID_COLUMN_NAME, RELATION_LABEL_COLUMN_NAME, RELATION_PATTERN_TYPES, _get_skyline

#: fixme: deprecated
SUBSET_LABELS = ("testing", "training", "validation", "total")


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


def _test_count_dataframe(
    df: pandas.DataFrame,
    id_column_name: str,
    label_column_name: str,
    labels: bool = False,
    total: bool = True,
):
    """Check the general structure of a count dataframe."""
    # check correct output type
    assert isinstance(df, pandas.DataFrame)

    expected_columns = {id_column_name, COUNT_COLUMN_NAME}
    if labels:
        expected_columns.add(label_column_name)
    if not total:
        expected_columns.add(SUBSET_COLUMN_NAME)

    # check columns
    assert expected_columns == set(df.columns)

    # check value range and type
    assert (df[COUNT_COLUMN_NAME] >= 0).all()
    assert df[COUNT_COLUMN_NAME].dtype == np.int64

    # check value range subset
    if total:
        assert set(SUBSET_LABELS).issuperset(df[SUBSET_COLUMN_NAME].unique())


class AnalysisTests(unittest.TestCase):
    """Tests for dataset analysis utilities."""

    def setUp(self) -> None:
        """Initialize the unittest."""
        self.dataset = Nations()

    def test_relation_count_dataframe(self):
        """Test relation count dataframe."""
        for labels, total in itertools.product((False, True), repeat=2):
            df = get_relation_count_df(dataset=self.dataset, add_labels=labels, total_count=total)
            _test_count_dataframe(
                df=df,
                id_column_name=RELATION_ID_COLUMN_NAME,
                label_column_name=RELATION_LABEL_COLUMN_NAME,
                labels=labels,
                total=total,
            )

    def test_entity_count_dataframe(self):
        """Test entity count dataframe."""
        for labels, total in itertools.product((False, True), repeat=2):
            df = get_entity_count_df(dataset=self.dataset, add_labels=labels, total_count=total)
            _test_count_dataframe(
                df=df,
                id_column_name=RELATION_ID_COLUMN_NAME,
                label_column_name=RELATION_LABEL_COLUMN_NAME,
                labels=labels,
                total=total,
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
