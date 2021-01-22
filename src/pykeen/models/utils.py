# -*- coding: utf-8 -*-

"""Utilities for models."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..typing import MappedTriples


def get_novelty_mask(
    mapped_triples: MappedTriples,
    query_ids: np.ndarray,
    col: int,
    other_col_ids: Tuple[int, int],
) -> np.ndarray:
    r"""Calculate for each query ID whether it is novel.

    In particular, computes:

    .. math ::
        q \notin \{t[col] in T \mid t[\neg col] = p\}

    for each q in query_ids where :math:`\neg col` denotes all columns but `col`, and `p` equals `other_col_ids`.

    :param mapped_triples: shape: (num_triples, 3), dtype: long
        The mapped triples (i.e. ID-based).
    :param query_ids: shape: (num_queries,), dtype: long
        The query IDs. Are assumed to be unique (i.e. without duplicates).
    :param col:
        The column to which the query IDs correspond.
    :param other_col_ids:
        Fixed IDs for the other columns.

    :return: shape: (num_queries,), dtype: bool
        A boolean mask indicating whether the ID does not correspond to a known triple.
    """
    other_cols = sorted(set(range(mapped_triples.shape[1])).difference({col}))
    other_col_ids = torch.tensor(data=other_col_ids, dtype=torch.long, device=mapped_triples.device)
    filter_mask = (mapped_triples[:, other_cols] == other_col_ids[None, :]).all(dim=-1)  # type: ignore
    known_ids = mapped_triples[filter_mask, col].unique().cpu().numpy()
    return np.isin(element=query_ids, test_elements=known_ids, assume_unique=True, invert=True)


def get_novelty_all_mask(
    mapped_triples: MappedTriples,
    query: np.ndarray,
) -> np.ndarray:
    """Get novelty mask."""
    known = {tuple(triple) for triple in mapped_triples.tolist()}
    return np.asarray(
        [tuple(triple) not in known for triple in query],
        dtype=np.bool,
    )


def _postprocess_prediction_df(
    rv: pd.DataFrame,
    *,
    col: int,
    add_novelties: bool,
    remove_known: bool,
    training: Optional[torch.LongTensor],
    testing: Optional[torch.LongTensor],
    query_ids_key: str,
    other_col_ids: Tuple[int, int],
) -> pd.DataFrame:
    if add_novelties or remove_known:
        rv['in_training'] = ~get_novelty_mask(
            mapped_triples=training,
            query_ids=rv[query_ids_key],
            col=col,
            other_col_ids=other_col_ids,
        )
    if add_novelties and testing is not None:
        rv['in_testing'] = ~get_novelty_mask(
            mapped_triples=testing,
            query_ids=rv[query_ids_key],
            col=col,
            other_col_ids=other_col_ids,
        )
    return _process_remove_known(rv, remove_known, testing)


def _postprocess_prediction_all_df(
    df: pd.DataFrame,
    *,
    add_novelties: bool,
    remove_known: bool,
    training: Optional[torch.LongTensor],
    testing: Optional[torch.LongTensor],
) -> pd.DataFrame:
    if add_novelties or remove_known:
        assert training is not None
        df['in_training'] = ~get_novelty_all_mask(
            mapped_triples=training,
            query=df[['head_id', 'relation_id', 'tail_id']].values,
        )
    if add_novelties and testing is not None:
        assert testing is not None
        df['in_testing'] = ~get_novelty_all_mask(
            mapped_triples=testing,
            query=df[['head_id', 'relation_id', 'tail_id']].values,
        )
    return _process_remove_known(df, remove_known, testing)


def _process_remove_known(df: pd.DataFrame, remove_known: bool, testing: Optional[torch.LongTensor]) -> pd.DataFrame:
    if not remove_known:
        return df

    df = df[~df['in_training']]
    del df['in_training']
    if testing is None:
        return df

    df = df[~df['in_testing']]
    del df['in_testing']
    return df
