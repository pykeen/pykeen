# -*- coding: utf-8 -*-

"""Prediction workflows."""

import itertools as itt
import logging
from typing import Optional, Tuple, Union

import pandas as pd
import torch

from .base import Model
from .utils import _postprocess_prediction_all_df, _postprocess_prediction_df

logger = logging.getLogger(__name__)


def predict_scores(model: Model, triples: torch.LongTensor) -> torch.FloatTensor:
    """Calculate the scores for triples.

    This method takes head, relation and tail of each triple and calculates the corresponding score.

    Additionally, the model is set to evaluation mode.

    :param triples: shape: (number of triples, 3), dtype: long
        The indices of (head, relation, tail) triples.

    :return: shape: (number of triples, 1), dtype: float
        The score for each triple.
    """
    # Enforce evaluation mode
    model.eval()
    scores = model.score_hrt(triples)
    if model.predict_with_sigmoid:
        scores = torch.sigmoid(scores)
    return scores


def predict_heads(
    model: Model,
    relation_label: str,
    tail_label: str,
    add_novelties: bool = True,
    remove_known: bool = False,
    testing: Optional[torch.LongTensor] = None,
) -> pd.DataFrame:
    """Predict tails for the given head and relation (given by label).

    :param relation_label: The string label for the relation
    :param tail_label: The string label for the tail entity
    :param add_novelties: Should the dataframe include a column denoting if the ranked head entities correspond
     to novel triples?
    :param remove_known: Should non-novel triples (those appearing in the training set) be shown with the results?
     On one hand, this allows you to better assess the goodness of the predictions - you want to see that the
     non-novel triples generally have higher scores. On the other hand, if you're doing hypothesis generation, they
     may pose as a distraction. If this is set to True, then non-novel triples will be removed and the column
     denoting novelty will be excluded, since all remaining triples will be novel. Defaults to false.
    :param testing: The mapped_triples from the testing triples factory (TriplesFactory.mapped_triples)

    The following example shows that after you train a model on the Nations dataset,
    you can score all entities w.r.t a given relation and tail entity.

    >>> from pykeen.pipeline import pipeline
    >>> result = pipeline(
    ...     dataset='Nations',
    ...     model='RotatE',
    ... )
    >>> df = result.model.predict_heads('accusation', 'brazil')
    """
    tail_id = model.triples_factory.entity_to_id[tail_label]
    relation_id = model.triples_factory.relation_to_id[relation_label]
    rt_batch = torch.tensor([[relation_id, tail_id]], dtype=torch.long, device=model.device)
    scores = model.predict_scores_all_heads(rt_batch)
    scores = scores[0, :].tolist()
    rv = pd.DataFrame(
        [
            (entity_id, entity_label, scores[entity_id])
            for entity_label, entity_id in model.triples_factory.entity_to_id.items()
        ],
        columns=['head_id', 'head_label', 'score'],
    ).sort_values('score', ascending=False)

    return _postprocess_prediction_df(
        rv=rv,
        add_novelties=add_novelties,
        remove_known=remove_known,
        training=model.triples_factory.mapped_triples,
        testing=testing,
        query_ids_key='head_id',
        col=0,
        other_col_ids=(relation_id, tail_id),
    )


def predict_tails(
    model: Model,
    head_label: str,
    relation_label: str,
    add_novelties: bool = True,
    remove_known: bool = False,
    testing: Optional[torch.LongTensor] = None,
) -> pd.DataFrame:
    """Predict tails for the given head and relation (given by label).

    :param head_label: The string label for the head entity
    :param relation_label: The string label for the relation
    :param add_novelties: Should the dataframe include a column denoting if the ranked tail entities correspond
     to novel triples?
    :param remove_known: Should non-novel triples (those appearing in the training set) be shown with the results?
     On one hand, this allows you to better assess the goodness of the predictions - you want to see that the
     non-novel triples generally have higher scores. On the other hand, if you're doing hypothesis generation, they
     may pose as a distraction. If this is set to True, then non-novel triples will be removed and the column
     denoting novelty will be excluded, since all remaining triples will be novel. Defaults to false.
    :param testing: The mapped_triples from the testing triples factory (TriplesFactory.mapped_triples)

    The following example shows that after you train a model on the Nations dataset,
    you can score all entities w.r.t a given head entity and relation.

    >>> from pykeen.pipeline import pipeline
    >>> result = pipeline(
    ...     dataset='Nations',
    ...     model='RotatE',
    ... )
    >>> df = result.model.predict_tails('brazil', 'accusation')
    """
    head_id = model.triples_factory.entity_to_id[head_label]
    relation_id = model.triples_factory.relation_to_id[relation_label]
    batch = torch.tensor([[head_id, relation_id]], dtype=torch.long, device=model.device)
    scores = model.predict_scores_all_tails(batch)
    scores = scores[0, :].tolist()
    rv = pd.DataFrame(
        [
            (entity_id, entity_label, scores[entity_id])
            for entity_label, entity_id in model.triples_factory.entity_to_id.items()
        ],
        columns=['tail_id', 'tail_label', 'score'],
    ).sort_values('score', ascending=False)

    return _postprocess_prediction_df(
        rv,
        add_novelties=add_novelties,
        remove_known=remove_known,
        testing=testing,
        training=model.triples_factory.mapped_triples,
        query_ids_key='tail_id',
        col=2,
        other_col_ids=(head_id, relation_id),
    )


def predict_scores_all_relations(
    model: Model,
    ht_batch: torch.LongTensor,
    slice_size: Optional[int] = None,
) -> torch.FloatTensor:
    """Forward pass using middle (relation) prediction for obtaining scores of all possible relations.

    This method calculates the score for all possible relations for each (head, tail) pair.

    Additionally, the model is set to evaluation mode.

    :param ht_batch: shape: (batch_size, 2), dtype: long
        The indices of (head, tail) pairs.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.

    :return: shape: (batch_size, num_relations), dtype: float
        For each h-t pair, the scores for all possible relations.
    """
    # Enforce evaluation mode
    model.eval()
    if slice_size is None:
        scores = model.score_r(ht_batch)
    else:
        scores = model.score_r(ht_batch, slice_size=slice_size)
    if model.predict_with_sigmoid:
        scores = torch.sigmoid(scores)
    return scores


def score_all_triples(
    model: Model,
    k: Optional[int] = None,
    batch_size: int = 1,
    return_tensors: bool = False,
    add_novelties: bool = True,
    remove_known: bool = False,
    testing: Optional[torch.LongTensor] = None,
) -> Union[Tuple[torch.LongTensor, torch.FloatTensor], pd.DataFrame]:
    """Compute scores for all triples, optionally returning only the k highest scoring.

    .. note:: This operation is computationally very expensive for reasonably-sized knowledge graphs.
    .. warning:: Setting k=None may lead to huge memory requirements.

    :param k:
        The number of triples to return. Set to None, to keep all.

    :param batch_size:
        The batch size to use for calculating scores.

    :return: shape: (k, 3)
        A tensor containing the k highest scoring triples, or all possible triples if k=None.

    Example usage:

    .. code-block:: python

        from pykeen.pipeline import pipeline

        # Train a model (quickly)
        result = pipeline(model='RotatE', dataset='Nations', training_kwargs=dict(num_epochs=5))
        model = result.model

        # Get scores for *all* triples
        tensor = model.score_all_triples()
        df = model.make_labeled_df(tensor)

        # Get scores for top 15 triples
        top_df = model.score_all_triples(k=15)
    """
    # set model to evaluation mode
    model.eval()

    # Do not track gradients
    with torch.no_grad():
        logger.warning(
            f'score_all_triples is an expensive operation, involving {model.num_entities ** 2 * model.num_relations} '
            f'score evaluations.',
        )

        if k is None:
            logger.warning(
                'Not providing k to score_all_triples entails huge memory requirements for reasonably-sized '
                'knowledge graphs.',
            )
            return _score_all_triples(
                model=model,
                batch_size=batch_size,
                return_tensors=return_tensors,
                testing=testing,
                add_novelties=add_novelties,
                remove_known=remove_known,
            )

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
            top_scores = model.predict_scores_all_tails(hr_batch=hr_batch).view(-1)

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

        # Sort final result
        scores, indices = torch.sort(scores, descending=True)
        result = result[indices]

    if return_tensors:
        return result, scores

    rv = model.triples_factory.tensor_to_df(result, score=scores)
    return _postprocess_prediction_all_df(
        df=rv,
        add_novelties=add_novelties,
        remove_known=remove_known,
        training=model.triples_factory.mapped_triples,
        testing=testing,
    )


def _score_all_triples(
    model: Model,
    batch_size: int = 1,
    return_tensors: bool = False,
    *,
    add_novelties: bool = True,
    remove_known: bool = False,
    testing: Optional[torch.LongTensor] = None,
) -> Union[Tuple[torch.LongTensor, torch.FloatTensor], pd.DataFrame]:
    """Compute and store scores for all triples.

    :return: Parallel arrays of triples and scores
    """
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
        scores[r, e:e + batch_size, :] = model.predict_scores_all_tails(hr_batch=hr_batch).to(scores.device)

    # Explicitly create triples
    triples = torch.stack([
        torch.arange(model.num_relations).view(-1, 1, 1).repeat(1, model.num_entities, model.num_entities),
        torch.arange(model.num_entities).view(1, -1, 1).repeat(model.num_relations, 1, model.num_entities),
        torch.arange(model.num_entities).view(1, 1, -1).repeat(model.num_relations, model.num_entities, 1),
    ], dim=-1).view(-1, 3)[:, [1, 0, 2]]

    # Sort final result
    scores, ind = torch.sort(scores.flatten(), descending=True)
    triples = triples[ind]

    if return_tensors:
        return triples, scores

    rv = model.triples_factory.tensor_to_df(triples, score=scores)
    return _postprocess_prediction_all_df(
        df=rv,
        add_novelties=add_novelties,
        remove_known=remove_known,
        training=model.triples_factory.mapped_triples,
        testing=testing,
    )
