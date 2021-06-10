# -*- coding: utf-8 -*-

"""Prediction workflows."""

import itertools as itt
import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .base import Model
from ..triples import CoreTriplesFactory, TriplesFactory
from ..typing import MappedTriples, ScorePack

__all__ = [
    'predict',
    'get_all_prediction_df',
    'get_head_prediction_df',
    'get_relation_prediction_df',
    'get_tail_prediction_df',
]

logger = logging.getLogger(__name__)


def get_head_prediction_df(
    model: Model,
    relation_label: str,
    tail_label: str,
    *,
    triples_factory: TriplesFactory,
    add_novelties: bool = True,
    remove_known: bool = False,
    testing: Optional[torch.LongTensor] = None,
) -> pd.DataFrame:
    """Predict heads for the given relation and tail (given by label).

    :param model: A PyKEEN model
    :param relation_label: The string label for the relation
    :param tail_label: The string label for the tail entity
    :param triples_factory: Training triples factory
    :param add_novelties: Should the dataframe include a column denoting if the ranked head entities correspond
        to novel triples?
    :param remove_known: Should non-novel triples (those appearing in the training set) be shown with the results?
        On one hand, this allows you to better assess the goodness of the predictions - you want to see that the
        non-novel triples generally have higher scores. On the other hand, if you're doing hypothesis generation, they
        may pose as a distraction. If this is set to True, then non-novel triples will be removed and the column
        denoting novelty will be excluded, since all remaining triples will be novel. Defaults to false.
    :param testing: The mapped_triples from the testing triples factory (TriplesFactory.mapped_triples)
    :return: shape: (k, 3)
        A dataframe with columns based on the settings or a tensor. Contains either the k highest scoring triples,
        or all possible triples if k is None.

    The following example shows that after you train a model on the Nations dataset,
    you can score all entities w.r.t a given relation and tail entity.

    >>> from pykeen.pipeline import pipeline
    >>> from pykeen.models.predict import get_head_prediction_df
    >>> result = pipeline(
    ...     dataset='Nations',
    ...     model='RotatE',
    ... )
    >>> df = get_head_prediction_df(result.model, 'accusation', 'brazil', triples_factory=result.training)
    """
    tail_id = triples_factory.entity_to_id[tail_label]
    relation_id = triples_factory.relation_to_id[relation_label]
    rt_batch = torch.as_tensor([[relation_id, tail_id]], dtype=torch.long, device=model.device)
    scores = model.predict_h(rt_batch)
    scores = scores[0, :].tolist()
    rv = pd.DataFrame(
        [
            (entity_id, entity_label, scores[entity_id])
            for entity_label, entity_id in triples_factory.entity_to_id.items()
        ],
        columns=['head_id', 'head_label', 'score'],
    ).sort_values('score', ascending=False)

    return _postprocess_prediction_df(
        df=rv,
        add_novelties=add_novelties,
        remove_known=remove_known,
        training=triples_factory.mapped_triples,
        testing=testing,
        query_ids_key='head_id',
        col=0,
        other_col_ids=(relation_id, tail_id),
    )


def get_tail_prediction_df(
    model: Model,
    head_label: str,
    relation_label: str,
    *,
    triples_factory: TriplesFactory,
    add_novelties: bool = True,
    remove_known: bool = False,
    testing: Optional[torch.LongTensor] = None,
) -> pd.DataFrame:
    """Predict tails for the given head and relation (given by label).

    :param model: A PyKEEN model
    :param head_label: The string label for the head entity
    :param relation_label: The string label for the relation
    :param triples_factory: Training triples factory
    :param add_novelties: Should the dataframe include a column denoting if the ranked tail entities correspond
        to novel triples?
    :param remove_known: Should non-novel triples (those appearing in the training set) be shown with the results?
        On one hand, this allows you to better assess the goodness of the predictions - you want to see that the
        non-novel triples generally have higher scores. On the other hand, if you're doing hypothesis generation, they
        may pose as a distraction. If this is set to True, then non-novel triples will be removed and the column
        denoting novelty will be excluded, since all remaining triples will be novel. Defaults to false.
    :param testing: The mapped_triples from the testing triples factory (TriplesFactory.mapped_triples)
    :return: shape: (k, 3)
        A dataframe with columns based on the settings or a tensor. Contains either the k highest scoring triples,
        or all possible triples if k is None.

    The following example shows that after you train a model on the Nations dataset,
    you can score all entities w.r.t a given head entity and relation.

    >>> from pykeen.pipeline import pipeline
    >>> from pykeen.models.predict import get_tail_prediction_df
    >>> result = pipeline(
    ...     dataset='Nations',
    ...     model='RotatE',
    ... )
    >>> df = get_tail_prediction_df(result.model, 'brazil', 'accusation', triples_factory=result.training)
    """
    head_id = triples_factory.entity_to_id[head_label]
    relation_id = triples_factory.relation_to_id[relation_label]
    batch = torch.as_tensor([[head_id, relation_id]], dtype=torch.long, device=model.device)
    scores = model.predict_t(batch)
    scores = scores[0, :].tolist()
    rv = pd.DataFrame(
        [
            (entity_id, entity_label, scores[entity_id])
            for entity_label, entity_id in triples_factory.entity_to_id.items()
        ],
        columns=['tail_id', 'tail_label', 'score'],
    ).sort_values('score', ascending=False)

    return _postprocess_prediction_df(
        rv,
        add_novelties=add_novelties,
        remove_known=remove_known,
        testing=testing,
        training=triples_factory.mapped_triples,
        query_ids_key='tail_id',
        col=2,
        other_col_ids=(head_id, relation_id),
    )


def get_relation_prediction_df(
    model: Model,
    head_label: str,
    tail_label: str,
    *,
    triples_factory: TriplesFactory,
    add_novelties: bool = True,
    remove_known: bool = False,
    testing: Optional[torch.LongTensor] = None,
) -> pd.DataFrame:
    """Predict relations for the given head and tail (given by label).

    :param model: A PyKEEN model
    :param head_label: The string label for the head entity
    :param tail_label: The string label for the tail entity
    :param triples_factory: Training triples factory
    :param add_novelties: Should the dataframe include a column denoting if the ranked relations correspond
        to novel triples?
    :param remove_known: Should non-novel triples (those appearing in the training set) be shown with the results?
        On one hand, this allows you to better assess the goodness of the predictions - you want to see that the
        non-novel triples generally have higher scores. On the other hand, if you're doing hypothesis generation, they
        may pose as a distraction. If this is set to True, then non-novel triples will be removed and the column
        denoting novelty will be excluded, since all remaining triples will be novel. Defaults to false.
    :param testing: The mapped_triples from the testing triples factory (TriplesFactory.mapped_triples)
    :return: shape: (k, 3)
        A dataframe with columns based on the settings or a tensor. Contains either the k highest scoring triples,
        or all possible triples if k is None.

    The following example shows that after you train a model on the Nations dataset,
    you can score all relations w.r.t a given head entity and tail entity.

    >>> from pykeen.pipeline import pipeline
    >>> from pykeen.models.predict import get_relation_prediction_df
    >>> result = pipeline(
    ...     dataset='Nations',
    ...     model='RotatE',
    ... )
    >>> df = get_relation_prediction_df(result.model, 'brazil', 'uk', triples_factory=result.training)
    """
    head_id = triples_factory.entity_to_id[head_label]
    tail_id = triples_factory.entity_to_id[tail_label]
    batch = torch.as_tensor([[head_id, tail_id]], dtype=torch.long, device=model.device)
    scores = model.predict_r(batch)
    scores = scores[0, :].tolist()
    rv = pd.DataFrame(
        [
            (relation_id, relation_label, scores[relation_id])
            for relation_label, relation_id in triples_factory.relation_to_id.items()
        ],
        columns=['relation_id', 'relation_label', 'score'],
    ).sort_values('score', ascending=False)

    return _postprocess_prediction_df(
        rv,
        add_novelties=add_novelties,
        remove_known=remove_known,
        testing=testing,
        training=triples_factory.mapped_triples,
        query_ids_key='relation_id',
        col=1,
        other_col_ids=(head_id, tail_id),
    )


def get_all_prediction_df(
    model: Model,
    *,
    triples_factory: CoreTriplesFactory,
    k: Optional[int] = None,
    batch_size: int = 1,
    return_tensors: bool = False,
    add_novelties: bool = True,
    remove_known: bool = False,
    testing: Optional[torch.LongTensor] = None,
) -> Union[ScorePack, pd.DataFrame]:
    """Compute scores for all triples, optionally returning only the k highest scoring.

    .. note:: This operation is computationally very expensive for reasonably-sized knowledge graphs.
    .. warning:: Setting k=None may lead to huge memory requirements.

    :param model: A PyKEEN model
    :param triples_factory: Training triples factory
    :param k: The number of triples to return. Set to ``None`` to keep all.
    :param batch_size: The batch size to use for calculating scores
    :param return_tensors: If true, only return tensors. If false (default), return as a pandas DataFrame
    :param add_novelties: Should the dataframe include a column denoting if the ranked relations correspond
        to novel triples?
    :param remove_known: Should non-novel triples (those appearing in the training set) be shown with the results?
        On one hand, this allows you to better assess the goodness of the predictions - you want to see that the
        non-novel triples generally have higher scores. On the other hand, if you're doing hypothesis generation, they
        may pose as a distraction. If this is set to True, then non-novel triples will be removed and the column
        denoting novelty will be excluded, since all remaining triples will be novel. Defaults to false.
    :param testing: The mapped_triples from the testing triples factory (TriplesFactory.mapped_triples)
    :return: shape: (k, 3)
        A dataframe with columns based on the settings or a tensor. Contains either the k highest scoring triples,
        or all possible triples if k is None.

    Example usage:

    .. code-block:: python

        from pykeen.pipeline import pipeline
        from pykeen.models.predict import get_all_prediction_df

        # Train a model (quickly)
        result = pipeline(model='RotatE', dataset='Nations', training_kwargs=dict(num_epochs=5))
        model = result.model

        # Get scores for *all* triples
        df = get_all_prediction_df(model, triples_factory=result.training)

        # Get scores for top 15 triples
        top_df = get_all_prediction_df(model, k=15, triples_factory=result.training)
    """
    score_pack = predict(model=model, k=k, batch_size=batch_size)
    if return_tensors:
        return score_pack

    df = triples_factory.tensor_to_df(score_pack.result, score=score_pack.scores)
    return _postprocess_prediction_all_df(
        df=df,
        add_novelties=add_novelties,
        remove_known=remove_known,
        training=triples_factory.mapped_triples,
        testing=testing,
    )


def predict(model: Model, *, k: Optional[int] = None, batch_size: int = 1) -> ScorePack:
    """Calculate and store scores for either all triples, or the top k triples.

    :param model: A PyKEEN model
    :param k: The number of triples to return. Set to ``None`` to keep all.
    :param batch_size: The batch size to use for calculating scores
    :return: A score pack of parallel triples and scores
    """
    logger.warning(
        f'_predict is an expensive operation, involving {model.num_entities ** 2 * model.num_relations} '
        f'score evaluations.',
    )

    if k is not None:
        return _predict_k(model=model, k=k, batch_size=batch_size)

    logger.warning(
        'Not providing k to score_all_triples entails huge memory requirements for reasonably-sized '
        'knowledge graphs.',
    )
    return _predict_all(model=model, batch_size=batch_size)


@torch.no_grad()
def _predict_all(model: Model, *, batch_size: int = 1) -> ScorePack:
    """Compute and store scores for all triples.

    :param model: A PyKEEN model
    :param batch_size: The batch size to use for calculating scores
    :return: A score pack of parallel triples and scores
    """
    model.eval()  # set model to evaluation mode

    # initialize buffer on cpu
    scores = torch.empty(model.num_relations, model.num_entities, model.num_entities, dtype=torch.float32)
    assert model.num_entities ** 2 * model.num_relations < (2 ** 63 - 1)

    for r, e in itt.product(
        range(model.num_relations),
        range(0, model.num_entities, batch_size),
    ):
        # calculate batch scores
        hs = torch.arange(e, min(e + batch_size, model.num_entities), device=model.device)
        hr_batch = torch.stack([
            hs,
            hs.new_empty(1).fill_(value=r).repeat(hs.shape[0]),
        ], dim=-1)
        scores[r, e:e + batch_size, :] = model.predict_t(hr_batch=hr_batch).to(scores.device)

    # Explicitly create triples
    result = torch.stack([
        torch.arange(model.num_relations).view(-1, 1, 1).repeat(1, model.num_entities, model.num_entities),
        torch.arange(model.num_entities).view(1, -1, 1).repeat(model.num_relations, 1, model.num_entities),
        torch.arange(model.num_entities).view(1, 1, -1).repeat(model.num_relations, model.num_entities, 1),
    ], dim=-1).view(-1, 3)[:, [1, 0, 2]]

    return _build_pack(result=result, scores=scores, flatten=True)


@torch.no_grad()
def _predict_k(model: Model, *, k: int, batch_size: int = 1) -> ScorePack:
    """Compute and store scores for the top k-scoring triples.

    :param model: A PyKEEN model
    :param k: The number of triples to return
    :param batch_size: The batch size to use for calculating scores
    :return: A score pack of parallel triples and scores
    """
    model.eval()  # set model to evaluation mode

    # initialize buffer on device
    result = torch.ones(0, 3, dtype=torch.long, device=model.device)
    scores = torch.empty(0, dtype=torch.float32, device=model.device)

    for r, e in itt.product(
        range(model.num_relations),
        range(0, model.num_entities, batch_size),
    ):
        # calculate batch scores
        hs = torch.arange(e, min(e + batch_size, model.num_entities), device=model.device)
        real_batch_size = hs.shape[0]
        hr_batch = torch.stack([
            hs,
            hs.new_empty(1).fill_(value=r).repeat(real_batch_size),
        ], dim=-1)
        top_scores = model.predict_t(hr_batch=hr_batch).view(-1)

        # get top scores within batch
        if top_scores.numel() >= k:
            top_scores, top_indices = top_scores.topk(k=min(k, batch_size), largest=True, sorted=False)
            top_heads, top_tails = top_indices // model.num_entities, top_indices % model.num_entities
        else:
            top_heads = hs.view(-1, 1).repeat(1, model.num_entities).view(-1)
            top_tails = torch.arange(model.num_entities, device=hs.device).view(1, -1).repeat(
                real_batch_size, 1).view(-1)

        top_triples = torch.stack([
            top_heads,
            top_heads.new_empty(top_heads.shape).fill_(value=r),
            top_tails,
        ], dim=-1)

        # append to global top scores
        scores = torch.cat([scores, top_scores])
        result = torch.cat([result, top_triples])

        # reduce size if necessary
        if result.shape[0] > k:
            scores, indices = scores.topk(k=k, largest=True, sorted=False)
            result = result[indices]

    return _build_pack(result=result, scores=scores)


def _build_pack(result: torch.LongTensor, scores: torch.FloatTensor, flatten: bool = False) -> ScorePack:
    """Sort final result and package in a score pack."""
    scores, indices = torch.sort(scores.flatten() if flatten else scores, descending=True)
    result = result[indices]
    return ScorePack(result=result, scores=scores)


def _postprocess_prediction_df(
    df: pd.DataFrame,
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
        df['in_training'] = ~get_novelty_mask(
            mapped_triples=training,
            query_ids=df[query_ids_key],
            col=col,
            other_col_ids=other_col_ids,
        )
    if add_novelties and testing is not None:
        df['in_testing'] = ~get_novelty_mask(
            mapped_triples=testing,
            query_ids=df[query_ids_key],
            col=col,
            other_col_ids=other_col_ids,
        )
    return _process_remove_known(df, remove_known, testing)


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
    other_col_ids = torch.as_tensor(data=other_col_ids, dtype=torch.long, device=mapped_triples.device)
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
        dtype=bool,
    )


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
