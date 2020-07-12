"""Tests for dataset analysis utilities."""
import itertools
import unittest
from typing import Collection, Optional, Sequence

import pandas

from pykeen.datasets import Nations
from pykeen.datasets.analysis import (
    SUBSET_LABELS,
    entity_count_dataframe,
    entity_relation_co_occurrence_dataframe,
    relation_count_dataframe,
)


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

    def test_relation_count_dataframe(self):
        """Test relation_count_dataframe()."""
        df = relation_count_dataframe(dataset=self.dataset)
        self._test_count_dataframe(
            df=df,
            label_name='relation_label',
            expected_labels=self.dataset.relation_to_id.keys(),
        )

    def test_entity_count_dataframe(self):
        """Test entity_count_dataframe()."""
        df = entity_count_dataframe(dataset=self.dataset)
        self._test_count_dataframe(
            df=df,
            label_name='entity_label',
            expected_labels=self.dataset.entity_to_id.keys(),
            second_level_column_labels=['head', 'tail', 'total'],
        )

    def test_entity_relation_co_occurrence_dataframe(self):
        """Test entity_relation_co_occurrence_dataframe()"""
        df = entity_relation_co_occurrence_dataframe(dataset=self.dataset)

        # check correct type
        assert isinstance(df, pandas.DataFrame)

        # check index
        self.assertListEqual(
            list(df.index),
            list(itertools.product(SUBSET_LABELS, sorted(self.dataset.entity_to_id.keys())))
        )
