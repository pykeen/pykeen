# -*- coding: utf-8 -*-

"""Prediction workflows."""

import itertools as itt
import logging
import warnings
from abc import abstractmethod
from operator import itemgetter
from typing import Collection, List, Optional, Sequence, Tuple, Union, cast

import numpy
import pandas as pd
import torch
from tqdm.auto import tqdm

from .base import Model
from ..constants import TARGET_TO_INDEX
from ..triples import CoreTriplesFactory, TriplesFactory
from ..triples.utils import tensor_to_df
from ..typing import (
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    InductiveMode,
    LabeledTriples,
    MappedTriples,
    ScorePack,
    Target,
)
from ..utils import is_cuda_oom_error

__all__ = [
    "predict",
    "predict_triples_df",
    "get_all_prediction_df",
    "get_prediction_df",
    # deprecated
    "get_head_prediction_df",
    "get_relation_prediction_df",
    "get_tail_prediction_df",
]

logger = logging.getLogger(__name__)


def _get_targets(
    ids: Union[None, torch.Tensor, Collection[str]],
    triples_factory: TriplesFactory,
    device: torch.device,
    entity: bool = True,
) -> Tuple[List[Tuple[str, int]], Optional[torch.Tensor]]:
    label_to_id = triples_factory.entity_to_id if entity else triples_factory.relation_to_id
    if ids is None:
        return sorted(label_to_id.items(), key=itemgetter(1)), None
    id_tensor = None
    if isinstance(ids, torch.Tensor):
        id_tensor = ids
        ids = ids.tolist()
    ids = triples_factory.entities_to_ids(entities=ids) if entity else triples_factory.relations_to_ids(relations=ids)
    ids = sorted(set(ids))
    if id_tensor is None:
        id_tensor = torch.as_tensor(ids, torch.long, device=device)
    id_to_label = triples_factory.entity_id_to_label if entity else triples_factory.relation_id_to_label
    return [(id_to_label[i], i) for i in ids], id_tensor


def _get_input_batch(
    triples_factory: TriplesFactory,
    # exactly one of them is None
    head_label: Optional[str] = None,
    relation_label: Optional[str] = None,
    tail_label: Optional[str] = None,
) -> Tuple[Target, torch.LongTensor, Tuple[int, int]]:
    """Prepare input batch for prediction.

    :param triples_factory:
        the triples factory used to translate labels to ids.
    :param head_label:
        the head entity label
    :param relation_label:
        the relation label
    :param tail_label:
        the tail entity label

    :raises ValueError:
        if not exactly one of {head_label, relation_label, tail_label} is None

    :return:
        a 3-tuple (target, batch, batch_tuple) of the prediction target, the input batch, and the input batch as tuple.
    """
    # create input batch
    batch_ids = []
    target = None
    if head_label:
        batch_ids.append(triples_factory.entity_to_id[head_label])
    else:
        target = LABEL_HEAD
    if relation_label:
        batch_ids.append(triples_factory.relation_to_id[relation_label])
    else:
        target = LABEL_RELATION
    if tail_label:
        batch_ids.append(triples_factory.entity_to_id[tail_label])
    else:
        target = LABEL_TAIL
    if target is None or len(batch_ids) != 2:
        raise ValueError(
            f"Exactly one of {{head,relation,tail}}_label must be None, but got "
            f"{head_label}, {relation_label}, {tail_label}",
        )

    batch = cast(torch.LongTensor, torch.as_tensor([batch_ids], dtype=torch.long))
    return target, batch, (batch_ids[0], batch_ids[1])


def _get_mapped_triples(mapped_triples: Union[CoreTriplesFactory, MappedTriples]):
    """Get mapped triples."""
    if isinstance(mapped_triples, CoreTriplesFactory):
        mapped_triples = mapped_triples.mapped_triples
    return mapped_triples


class PredictionPostProcessor:
    """A post-processor for predictions."""

    def __init__(self, **filter_triples: Union[None, CoreTriplesFactory, MappedTriples]) -> None:
        """Instantiate the processor.

        :param filter_triples:
            a mapping from keys to triples to be used for filtering. `None` entries will be ignored. The keys are
            used to derive column names.
        """
        self.filter_triples = {
            key: _get_mapped_triples(value) for key, value in filter_triples.items() if value is not None
        }

    @abstractmethod
    def _contains(self, df: pd.DataFrame, mapped_triples: MappedTriples, invert: bool = False) -> numpy.ndarray:
        """
        Return which of the rows of the given data frame are contained in the ID-based triples.

        :param df: nrows: n
            the predictions
        :param mapped_triples: shape: (m, 3)
            the ID-based triples
        :param invert:
            whether to invert the result

        :return: shape: (n,), dtype: bool
            a boolean mask indicating which row is contained in the given ID-based triples
        """
        raise NotImplementedError

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out known triples.

        .. note ::
            this operation does *not* work in-place.

        :param df:
            the predictions

        :return:
            the filtered dataframe
        """
        for mapped_triples in self.filter_triples.values():
            df = df[self._contains(df=df, mapped_triples=mapped_triples, invert=True)]
        return df

    def add_membership_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add columns indicating whether the triples are known.

        :param df:
            the predictions

        :return:
            the predictions with extra columns
        """
        for key, mapped_triples in self.filter_triples.items():
            df[f"in_{key}"] = self._contains(df=df, mapped_triples=mapped_triples)
        return df

    def process(self, df: pd.DataFrame, remove_known: bool, add_novelties: bool) -> pd.DataFrame:
        """
        Post-process a prediction dataframe.

        .. warning ::
            if both, `remove_known` and `add_novelties` are enabled, only the first will be applied.

        :param df:
            the dataframe of predictions
        :param remove_known:
            whether to remove rows corresponding to known triples
        :param add_novelties:
            whether to add extra columns denoting whether triples are novel given the filter triples

        :return:
            the filtered, modified or original predictions dataframe
        """
        if add_novelties and remove_known:
            logger.warning("Since remove_known is enabled, will not add novelty column")
            add_novelties = False
        if add_novelties:
            return self.add_membership_columns(df=df)
        if remove_known:
            return self.filter(df=df)
        return df


class SinglePredictionPostProcessor(PredictionPostProcessor):
    """Post-processor for single-target predictions."""

    def __init__(self, target: Target, other_columns_fixed_ids: Tuple[int, int], **kwargs) -> None:
        """Initialize the post-processor.

        :param target:
            the prediction target
        :param other_columns_fixed_ids:
            the fixed IDs for the other columns

        :param kwargs:
            additional keyword-based parameters passed to :meth:`PredictionPostProcessor.__init__`
        """
        super().__init__(**kwargs)
        self.target = target
        self.other_columns_fixed_ids = other_columns_fixed_ids

    # docstr-coverage: inherited
    def _contains(
        self, df: pd.DataFrame, mapped_triples: MappedTriples, invert: bool = False
    ) -> numpy.ndarray:  # noqa: D102
        col = TARGET_TO_INDEX[self.target]
        other_cols = sorted(set(range(mapped_triples.shape[1])).difference({col}))
        device = mapped_triples.device
        other_col_ids = torch.as_tensor(data=self.other_columns_fixed_ids, dtype=torch.long, device=device)
        filter_mask = (mapped_triples[:, other_cols] == other_col_ids[None, :]).all(dim=-1)
        known_ids = mapped_triples[filter_mask, col].unique()
        query_ids = torch.as_tensor(df[f"{self.target}_id"].to_numpy(), device=device)
        return torch.isin(elements=query_ids, test_elements=known_ids, assume_unique=True, invert=invert).cpu().numpy()


def isin_many_dim(elements: torch.Tensor, test_elements: torch.Tensor, dim: int = 0) -> torch.BoolTensor:
    """Return whether elements are contained in test elements."""
    inverse, counts = torch.cat([elements, test_elements], dim=dim).unique(
        return_counts=True, return_inverse=True, dim=dim
    )[1:]
    return counts[inverse[: elements.shape[dim]]] > 1


class AllPredictionPostProcessor(PredictionPostProcessor):
    """Post-processor for all-triples predictions."""

    # docstr-coverage: inherited
    def _contains(
        self, df: pd.DataFrame, mapped_triples: MappedTriples, invert: bool = False
    ) -> numpy.ndarray:  # noqa: D102
        contained = (
            isin_many_dim(
                elements=torch.as_tensor(
                    df[[f"{target}_id" for target, _ in sorted(TARGET_TO_INDEX.items(), key=itemgetter(1))]].values,
                    device=mapped_triples.device,
                ),
                test_elements=mapped_triples,
            )
            .cpu()
            .numpy()
        )
        if invert:
            return ~contained
        return contained


@torch.inference_mode()
def get_prediction_df(
    model: Model,
    triples_factory: TriplesFactory,
    *,
    # exactly one of them is None
    head_label: Optional[str] = None,
    relation_label: Optional[str] = None,
    tail_label: Optional[str] = None,
    #
    targets: Optional[Sequence[str]] = None,
    add_novelties: bool = True,
    remove_known: bool = False,
    testing: Optional[torch.LongTensor] = None,
    mode: Optional[InductiveMode] = None,
) -> pd.DataFrame:
    """Get predictions for the head, relation, and/or tail combination.

    .. note ::
        Exactly one of `head_label`, `relation_label` and `tail_label` should be None. This is the position
        which will be predicted.

    :param model:
        A PyKEEN model
    :param triples_factory:
        the training triples factory

    :param head_label:
        the head entity label. If None, predict heads
    :param relation_label:
        the relation label. If None, predict relations
    :param tail_label:
        the tail entity label. If None, predict tails
    :param targets:
        restrict prediction to these targets

    :param add_novelties:
        should the dataframe include a column denoting if the ranked head entities correspond to novel triples?
    :param remove_known:
        should non-novel triples (those appearing in the training set) be shown with the results?
        On one hand, this allows you to better assess the goodness of the predictions - you want to see that the
        non-novel triples generally have higher scores. On the other hand, if you're doing hypothesis generation, they
        may pose as a distraction. If this is set to True, then non-novel triples will be removed and the column
        denoting novelty will be excluded, since all remaining triples will be novel. Defaults to false.
    :param testing:
        the mapped_triples from the testing triples factory (TriplesFactory.mapped_triples)
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.

    :return: shape: (k, 3)
        A dataframe with columns based on the settings or a tensor. Contains either the k highest scoring triples,
        or all possible triples if k is None
    """
    # get input & target
    target, batch, other_col_ids = _get_input_batch(
        triples_factory, head_label=head_label, relation_label=relation_label, tail_label=tail_label
    )

    # get label-to-id mapping and prediction targets
    label_ids, targets = _get_targets(
        ids=targets, triples_factory=triples_factory, device=model.device, entity=relation_label is not None
    )

    # get scores
    scores = model.predict(batch, full_batch=False, mode=mode, ids=targets, target=target).squeeze(dim=0).tolist()

    # create raw dataframe
    rv = pd.DataFrame(
        [(target_id, target_label, score) for (target_label, target_id), score in zip(label_ids, scores)],
        columns=[f"{target}_id", f"{target}_label", "score"],
    ).sort_values("score", ascending=False)

    # postprocess prediction df
    return SinglePredictionPostProcessor(
        target=target, other_columns_fixed_ids=other_col_ids, training=triples_factory, testing=testing
    ).process(df=rv, remove_known=remove_known, add_novelties=add_novelties)


def get_head_prediction_df(
    model: Model,
    triples_factory: TriplesFactory,
    relation_label: str,
    tail_label: str,
    *,
    heads: Optional[Sequence[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Predict heads for the given relation and tail (given by label).

    :param model:
        A PyKEEN model
    :param triples_factory:
        the training triples factory

    :param relation_label:
        the string label for the relation
    :param tail_label:
        the string label for the tail entity
    :param heads:
        restrict head prediction to the given entities
    :param kwargs:
        additional keyword-based parameters passed to :func:`get_prediction_df`.
    :return: shape: (k, 3)
        A dataframe for head predictions. Contains either the k highest scoring triples,
        or all possible triples if k is None

    The following example shows that after you train a model on the Nations dataset,
    you can score all entities w.r.t. a given relation and tail entity.

    >>> from pykeen.pipeline import pipeline
    >>> from pykeen.models.predict import get_head_prediction_df
    >>> result = pipeline(
    ...     dataset='Nations',
    ...     model='RotatE',
    ... )
    >>> df = get_head_prediction_df(result.model, 'accusation', 'brazil', triples_factory=result.training)
    """
    warnings.warn("Please directly use `pykeen.models.predict.get_prediction_df`", DeprecationWarning)
    return get_prediction_df(
        model=model,
        triples_factory=triples_factory,
        relation_label=relation_label,
        tail_label=tail_label,
        targets=heads,
        **kwargs,
    )


def get_tail_prediction_df(
    model: Model,
    triples_factory: TriplesFactory,
    head_label: str,
    relation_label: str,
    *,
    tails: Optional[Sequence[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Predict tails for the given head and relation (given by label).

    :param model:
        A PyKEEN model
    :param triples_factory:
        the training triples factory

    :param head_label:
        the string label for the head entity
    :param relation_label:
        the string label for the relation
    :param tails:
        restrict tail prediction to the given entities
    :param kwargs:
        additional keyword-based parameters passed to :func:`get_prediction_df`.
    :return: shape: (k, 3)
        A dataframe for tail predictions. Contains either the k highest scoring triples,
        or all possible triples if k is None

    The following example shows that after you train a model on the Nations dataset,
    you can score all entities w.r.t. a given head entity and relation.

    >>> from pykeen.pipeline import pipeline
    >>> from pykeen.models.predict import get_tail_prediction_df
    >>> result = pipeline(
    ...     dataset='Nations',
    ...     model='RotatE',
    ... )
    >>> df = get_tail_prediction_df(result.model, 'brazil', 'accusation', triples_factory=result.training)

    The optional `tails` parameter can be used to restrict prediction to a subset of entities, e.g.
    >>> df = get_tail_prediction_df(
    ...     result.model,
    ...     'brazil',
    ...     'accusation',
    ...     triples_factory=result.training,
    ...     tails=["burma", "china", "india", "indonesia"],
    ... )
    """
    warnings.warn("Please directly use `pykeen.models.predict.get_prediction_df`", DeprecationWarning)
    return get_prediction_df(
        model=model,
        triples_factory=triples_factory,
        head_label=head_label,
        relation_label=relation_label,
        targets=tails,
        **kwargs,
    )


def get_relation_prediction_df(
    model: Model,
    triples_factory: TriplesFactory,
    head_label: str,
    tail_label: str,
    *,
    relations: Optional[Sequence[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Predict relations for the given head and tail (given by label).

    :param model:
        A PyKEEN model
    :param triples_factory:
        the training triples factory

    :param head_label:
        the string label for the head entity
    :param tail_label:
        the string label for the tail entity
    :param relations:
        restrict relation prediction to the given relations
    :param kwargs:
        additional keyword-based parameters passed to :func:`get_prediction_df`.
    :return: shape: (k, 3)
        A dataframe for relation predictions. Contains either the k highest scoring triples,
        or all possible triples if k is None

    The following example shows that after you train a model on the Nations dataset,
    you can score all relations w.r.t. a given head entity and tail entity.

    >>> from pykeen.pipeline import pipeline
    >>> from pykeen.models.predict import get_relation_prediction_df
    >>> result = pipeline(
    ...     dataset='Nations',
    ...     model='RotatE',
    ... )
    >>> df = get_relation_prediction_df(result.model, 'brazil', 'uk', triples_factory=result.training)
    """
    warnings.warn("Please directly use `pykeen.models.predict.get_prediction_df`", DeprecationWarning)
    return get_prediction_df(
        model=model,
        triples_factory=triples_factory,
        head_label=head_label,
        tail_label=tail_label,
        targets=relations,
        **kwargs,
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

    return AllPredictionPostProcessor(training=triples_factory, testing=testing).process(
        df=triples_factory.tensor_to_df(score_pack.result, score=score_pack.scores),
        remove_known=remove_known,
        add_novelties=add_novelties,
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
