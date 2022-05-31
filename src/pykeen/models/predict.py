# -*- coding: utf-8 -*-

"""Prediction workflows."""

import itertools as itt
import logging
from abc import abstractmethod
from typing import Optional, Sequence, Tuple, Union

import numpy
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .base import Model
from ..triples import CoreTriplesFactory, TriplesFactory
from ..triples.utils import tensor_to_df
from ..typing import InductiveMode, LabeledTriples, MappedTriples, ScorePack
from ..utils import is_cuda_oom_error, triple_tensor_to_set

__all__ = [
    "predict",
    "predict_triples_df",
    "get_all_prediction_df",
    "get_head_prediction_df",
    "get_relation_prediction_df",
    "get_tail_prediction_df",
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
    mode: Optional[InductiveMode] = None,
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
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.
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
    scores = model.predict_h(rt_batch, mode=mode)
    scores = scores[0, :].tolist()
    rv = pd.DataFrame(
        [
            (entity_id, entity_label, scores[entity_id])
            for entity_label, entity_id in triples_factory.entity_to_id.items()
        ],
        columns=["head_id", "head_label", "score"],
    ).sort_values("score", ascending=False)

    return _postprocess_prediction_df(
        df=rv,
        add_novelties=add_novelties,
        remove_known=remove_known,
        training=triples_factory.mapped_triples,
        testing=testing,
        query_ids_key="head_id",
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
    mode: Optional[InductiveMode] = None,
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
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.
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
    scores = model.predict_t(batch, mode=mode)
    scores = scores[0, :].tolist()
    rv = pd.DataFrame(
        [
            (entity_id, entity_label, scores[entity_id])
            for entity_label, entity_id in triples_factory.entity_to_id.items()
        ],
        columns=["tail_id", "tail_label", "score"],
    ).sort_values("score", ascending=False)

    return _postprocess_prediction_df(
        rv,
        add_novelties=add_novelties,
        remove_known=remove_known,
        testing=testing,
        training=triples_factory.mapped_triples,
        query_ids_key="tail_id",
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
    mode: Optional[InductiveMode] = None,
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
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.
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
    scores = model.predict_r(batch, mode=mode)
    scores = scores[0, :].tolist()
    rv = pd.DataFrame(
        [
            (relation_id, relation_label, scores[relation_id])
            for relation_label, relation_id in triples_factory.relation_to_id.items()
        ],
        columns=["relation_id", "relation_label", "score"],
    ).sort_values("score", ascending=False)

    return _postprocess_prediction_df(
        rv,
        add_novelties=add_novelties,
        remove_known=remove_known,
        testing=testing,
        training=triples_factory.mapped_triples,
        query_ids_key="relation_id",
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
    mode: Optional[InductiveMode] = None,
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
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.
    :return: shape: (k, 3)
        A dataframe with columns based on the settings or a tensor. Contains either the k highest scoring triples,
        or all possible triples if k is None.

    Example usage:

    .. code-block::

        from pykeen.pipeline import pipeline
        from pykeen.models.predict import get_all_prediction_df

        # Train a model (quickly)
        result = pipeline(model='RotatE', dataset='Nations', epochs=5)
        model = result.model

        # Get scores for *all* triples
        df = get_all_prediction_df(model, triples_factory=result.training)

        # Get scores for top 15 triples
        top_df = get_all_prediction_df(model, k=15, triples_factory=result.training)
    """
    score_pack = predict(model=model, k=k, batch_size=batch_size, mode=mode)
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


def predict(
    model: Model, *, k: Optional[int] = None, batch_size: int = 1, mode: Optional[InductiveMode] = None
) -> ScorePack:
    """Calculate and store scores for either all triples, or the top k triples.

    :param model: A PyKEEN model
    :param k: The number of triples to return. Set to ``None`` to keep all.
    :param batch_size: The batch size to use for calculating scores
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.
    :return: A score pack of parallel triples and scores
    """
    logger.warning(
        f"_predict is an expensive operation, involving {model.num_entities ** 2 * model.num_real_relations} "
        f"score evaluations.",
    )

    if k is not None:
        return _predict_k(model=model, k=k, batch_size=batch_size, mode=mode)

    logger.warning(
        "Not providing k to score_all_triples entails huge memory requirements for reasonably-sized "
        "knowledge graphs.",
    )
    return _predict_all(model=model, batch_size=batch_size, mode=mode)


class _ScoreConsumer:
    """A consumer of scores for visitor pattern."""

    result: torch.LongTensor
    scores: torch.FloatTensor
    flatten: bool

    @abstractmethod
    def __call__(
        self,
        head_id_range: Tuple[int, int],
        relation_id: int,
        hr_batch: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> None:
        """Consume scores for the given hr_batch."""
        raise NotImplementedError

    def finalize(self) -> ScorePack:
        """Finalize the result to build a score pack."""
        return _build_pack(result=self.result, scores=self.scores, flatten=self.flatten)


class _TopKScoreConsumer(_ScoreConsumer):
    """Collect top-k triples & scores."""

    flatten = False

    def __init__(self, k: int, device: torch.device) -> None:
        """
        Initialize the consumer.

        :param k:
            the number of top-scored triples to collect
        :param device:
            the model's device
        """
        self.k = k
        # initialize buffer on device
        self.result = torch.empty(0, 3, dtype=torch.long, device=device)
        self.scores = torch.empty(0, device=device)

    # docstr-coverage: inherited
    def __call__(
        self,
        head_id_range: Tuple[int, int],
        relation_id: int,
        hr_batch: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> None:  # noqa: D102
        batch_size, num_entities = scores.shape

        # reshape, shape: (batch_size * num_entities,)
        top_scores = scores.view(-1)

        # get top scores within batch
        if top_scores.numel() >= self.k:
            top_scores, top_indices = top_scores.topk(
                k=min(self.k, top_scores.numel()),
                largest=True,
                sorted=False,
            )
            h_start = head_id_range[0]
            top_heads = h_start + torch.div(top_indices, num_entities, rounding_mode="trunc")
            top_tails = top_indices % num_entities
        else:
            top_heads = hr_batch[:, 0].view(-1, 1).repeat(1, num_entities).view(-1)
            top_tails = torch.arange(num_entities, device=hr_batch.device).view(1, -1).repeat(batch_size, 1).view(-1)

        top_triples = torch.stack(
            [
                top_heads,
                top_heads.new_full(size=top_heads.shape, fill_value=relation_id, dtype=top_heads.dtype),
                top_tails,
            ],
            dim=-1,
        )

        # append to global top scores
        self.scores = torch.cat([self.scores, top_scores])
        self.result = torch.cat([self.result, top_triples])

        # reduce size if necessary
        if self.result.shape[0] > self.k:
            self.scores, indices = self.scores.topk(k=self.k, largest=True, sorted=False)
            self.result = self.result[indices]


class _AllConsumer(_ScoreConsumer):
    """Collect scores for all triples."""

    flatten = True

    def __init__(self, num_entities: int, num_relations: int) -> None:
        """
        Initialize the consumer.

        :param num_entities:
            the number of entities
        :param num_relations:
            the number of relations
        """
        assert num_entities**2 * num_relations < (2**63 - 1)
        # initialize buffer on cpu
        self.scores = torch.empty(num_relations, num_entities, num_entities, device="cpu")
        # Explicitly create triples
        self.result = torch.stack(
            [
                torch.arange(num_relations).view(-1, 1, 1).repeat(1, num_entities, num_entities),
                torch.arange(num_entities).view(1, -1, 1).repeat(num_relations, 1, num_entities),
                torch.arange(num_entities).view(1, 1, -1).repeat(num_relations, num_entities, 1),
            ],
            dim=-1,
        ).view(-1, 3)[:, [1, 0, 2]]

    # docstr-coverage: inherited
    def __call__(
        self,
        head_id_range: Tuple[int, int],
        relation_id: int,
        hr_batch: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> None:  # noqa: D102
        h_start, h_stop = head_id_range
        self.scores[relation_id, h_start:h_stop, :] = scores.to(self.scores.device)


@torch.inference_mode()
def _consume_scores(
    model: Model, *consumers: _ScoreConsumer, batch_size: int = 1, mode: Optional[InductiveMode]
) -> None:
    """
    Batch-wise calculation of all triple scores and consumption.

    :param model:
        the model, will be set to evaluation mode
    :param consumers:
        the consumers of score batches
    :param batch_size:
        the batch size to use  # TODO: automatic batch size maximization
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.
    """
    # TODO: in the future, we may want to expose this method
    # set model to evaluation mode
    model.eval()

    for r, h_start in tqdm(
        itt.product(
            range(model.num_real_relations),
            range(0, model.num_entities, batch_size),
        ),
        desc="scoring",
        unit="batch",
        unit_scale=True,
        total=model.num_relations * model.num_entities // batch_size,
    ):
        # calculate batch scores
        h_stop = min(h_start + batch_size, model.num_entities)
        hs = torch.arange(h_start, h_stop, device=model.device)
        hr_batch = torch.stack(
            [
                hs,
                hs.new_full(size=(hs.shape[0],), fill_value=r),
            ],
            dim=-1,
        )
        scores = model.predict_t(hr_batch=hr_batch, mode=mode)
        for consumer in consumers:
            consumer(head_id_range=(h_start, h_stop), relation_id=r, hr_batch=hr_batch, scores=scores)


@torch.inference_mode()
def _predict_all(model: Model, *, batch_size: int = 1, mode: Optional[InductiveMode]) -> ScorePack:
    """Compute and store scores for all triples.

    :param model: A PyKEEN model
    :param batch_size: The batch size to use for calculating scores
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.
    :return: A score pack of parallel triples and scores
    """
    consumer = _AllConsumer(num_entities=model.num_entities, num_relations=model.num_relations)
    _consume_scores(model, consumer, batch_size=batch_size, mode=mode)
    return consumer.finalize()


@torch.inference_mode()
def _predict_k(model: Model, *, k: int, batch_size: int = 1, mode: Optional[InductiveMode]) -> ScorePack:
    """Compute and store scores for the top k-scoring triples.

    :param model: A PyKEEN model
    :param k: The number of triples to return
    :param batch_size: The batch size to use for calculating scores
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.
    :return: A score pack of parallel triples and scores
    """
    consumer = _TopKScoreConsumer(k=k, device=model.device)
    _consume_scores(model, consumer, batch_size=batch_size, mode=mode)
    return consumer.finalize()


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
        df["in_training"] = ~get_novelty_mask(
            mapped_triples=training,
            query_ids=df[query_ids_key],
            col=col,
            other_col_ids=other_col_ids,
        )
    if add_novelties and testing is not None:
        df["in_testing"] = ~get_novelty_mask(
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
        df["in_training"] = ~get_novelty_all_mask(
            mapped_triples=training,
            query=df[["head_id", "relation_id", "tail_id"]].values,
        )
    if add_novelties and testing is not None:
        assert testing is not None
        df["in_testing"] = ~get_novelty_all_mask(
            mapped_triples=testing,
            query=df[["head_id", "relation_id", "tail_id"]].values,
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
    known = triple_tensor_to_set(mapped_triples)
    return np.asarray(
        [tuple(triple) not in known for triple in query],
        dtype=bool,
    )


def _process_remove_known(df: pd.DataFrame, remove_known: bool, testing: Optional[torch.LongTensor]) -> pd.DataFrame:
    if not remove_known:
        return df

    df = df[~df["in_training"]]
    del df["in_training"]
    if testing is None:
        return df

    df = df[~df["in_testing"]]
    del df["in_testing"]
    return df


@torch.inference_mode()
def _predict_triples(
    model: Model,
    mapped_triples: MappedTriples,
    batch_size: Optional[int] = None,
    *,
    mode: Optional[InductiveMode],
) -> torch.FloatTensor:
    """Predict scores for triples while dealing with reducing batch size for CUDA OOM."""
    # base case: infer maximum batch size
    if batch_size is None:
        return _predict_triples(
            model=model, mapped_triples=mapped_triples, batch_size=mapped_triples.shape[0], mode=mode
        )

    # base case: single batch
    if batch_size >= mapped_triples.shape[0]:
        return model.predict_hrt(hrt_batch=mapped_triples, mode=mode)

    if batch_size <= 0:
        # TODO: this could happen because of AMO
        raise ValueError("batch_size must be positive.")

    try:
        return torch.cat(
            [
                model.predict_hrt(hrt_batch=hrt_batch, mode=mode)
                for hrt_batch in mapped_triples.split(split_size=batch_size, dim=0)
            ],
            dim=0,
        )
    except RuntimeError as error:
        # TODO: Can we make AMO code re-usable? e.g. like https://gist.github.com/mberr/c37a8068b38cabc98228db2cbe358043
        if is_cuda_oom_error(error):
            return _predict_triples(model=model, mapped_triples=mapped_triples, batch_size=batch_size // 2)

        # no OOM error.
        raise error


def predict_triples_df(
    model: Model,
    *,
    triples: Union[None, MappedTriples, LabeledTriples, Union[Tuple[str, str, str], Sequence[Tuple[str, str, str]]]],
    triples_factory: Optional[CoreTriplesFactory] = None,
    batch_size: Optional[int] = None,
    mode: Optional[InductiveMode] = None,
) -> pd.DataFrame:
    """
    Predict on labeled or mapped triples.

    :param model:
        The model.
    :param triples: shape: (num_triples, 3)
        The triples in one of the following formats:

        - A single label-based triple.
        - A list of label-based triples.
        - An array of label-based triples
        - An array of ID-based triples.
        - None. In this case, a triples factory has to be provided, and its triples will be used.

    :param triples_factory:
        The triples factory. Must be given if triples are label-based. If provided and triples are ID-based, add labels
        to result.
    :param batch_size:
        The batch size to use. Use None for automatic memory optimization.
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.

    :return: columns: head_id | relation_id | tail_id | score | *
        A dataframe with one row per triple.

    :raises ValueError:
        If label-based triples have been provided, but the triples factory does not provide a mapping.

    The TransE model can be trained and used to predict a given triple.

    >>> from pykeen.pipeline import pipeline
    >>> result = pipeline(dataset="nations", model="TransE")
    >>> from pykeen.models.predict import predict_triples_df
    >>> df = predict_triples_df(
    ...     model=result.model,
    ...     triples=("uk", "conferences", "brazil"),
    ...     triples_factory=result.training,
    ... )
    """
    if triples is None:
        if triples_factory is None:
            raise ValueError("If no triples are provided, a triples_factory must be provided.")

        triples = triples_factory.mapped_triples

    if not isinstance(triples, torch.Tensor) or triples.dtype != torch.long:
        if triples_factory is None or not isinstance(triples_factory, TriplesFactory):
            raise ValueError("If triples are not ID-based, a triples_factory must be provided and label-based.")

        # make sure triples are a numpy array
        triples = numpy.asanyarray(triples)

        # make sure triples are 2d
        triples = numpy.atleast_2d(triples)

        # convert to ID-based
        triples = triples_factory.map_triples(triples)

    assert torch.is_tensor(triples) and triples.dtype == torch.long

    scores = _predict_triples(model=model, mapped_triples=triples, batch_size=batch_size, mode=mode).squeeze(dim=1)

    if triples_factory is None:
        return tensor_to_df(tensor=triples, score=scores)

    return triples_factory.tensor_to_df(tensor=triples, score=scores)
