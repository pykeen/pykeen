# -*- coding: utf-8 -*-

"""
Prediction workflows.

Train Model

>>> from pykeen.pipeline import pipeline
>>> from pykeen.models.predict import predict

>>> result = pipeline(dataset="nations", model="pairre", training_kwargs=dict(num_epochs=0))
>>> pack = predict(model=result.model)
>>> pred = pack.process(factory=result.training)
>>> pred_filtered = pred.filter_triples(result.training)
>>> pred_annotated = pred.add_membership_columns(training=result.training)
>>> pred_filtered.df
"""

import dataclasses
import logging
import math
import warnings
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Collection, List, Optional, Sequence, Tuple, Union, cast

import numpy
import pandas as pd
import torch
import torch.utils.data
from torch_max_mem import maximize_memory_utilization
from tqdm.auto import tqdm
from typing_extensions import TypeAlias  # Python <=3.9

from .base import Model
from ..constants import COLUMN_LABELS, TARGET_TO_INDEX
from ..triples import AnyTriples, CoreTriplesFactory, TriplesFactory, get_mapped_triples
from ..triples.utils import tensor_to_df
from ..typing import (  # ScorePack,
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    DeviceHint,
    InductiveMode,
    LabeledTriples,
    MappedTriples,
    Target,
)
from ..utils import resolve_device

__all__ = [
    "predict",
    "predict_triples_df",
    "get_all_prediction",
    "get_prediction_df",
    # score consumption / prediction loop
    "consume_scores",
    "ScoreConsumer",
    # deprecated
    "get_head_prediction_df",
    "get_relation_prediction_df",
    "get_tail_prediction_df",
]

logger = logging.getLogger(__name__)


# cf. https://github.com/python/mypy/issues/5374
@dataclasses.dataclass  # type: ignore
class Predictions(ABC):
    """Base class for predictions."""

    #: the dataframe; has to have a column named "score"
    df: pd.DataFrame

    #: an optional factory to use for labeling
    factory: Optional[CoreTriplesFactory]

    def __post_init__(self):
        """Verify constraints."""
        if "score" not in self.df.columns:
            raise ValueError(f"df must have a column named 'score', but df.columns={self.df.columns}")

    @classmethod
    def new(cls, df: pd.DataFrame, factory: Optional[CoreTriplesFactory]) -> "Predictions":
        """Create predictions with exchanged df/factory."""
        return cls(df=df, factory=factory)

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

    def filter_triples(self, *triples: Optional[AnyTriples]) -> pd.DataFrame:
        """Filter out known triples."""
        df = self.df
        for mapped_triples in triples:
            if mapped_triples is None:
                continue
            df = df[
                self._contains(
                    df=df, mapped_triples=get_mapped_triples(mapped_triples, factory=self.factory), invert=True
                )
            ]
        return self.new(df=df, factory=self.factory)

    def add_membership_columns(self, **filter_triples: Optional[AnyTriples]) -> pd.DataFrame:
        """Add columns indicating whether the triples are known."""
        df = self.df
        for key, mapped_triples in filter_triples.items():
            if mapped_triples is None:
                continue
            df[f"in_{key}"] = self._contains(
                df=df, mapped_triples=get_mapped_triples(mapped_triples, factory=self.factory)
            )
        return self.new(df=df, factory=self.factory)


@dataclasses.dataclass
class TriplePredictions(Predictions):
    """Triples with their predicted scores."""

    # predict_triples_df(triples) -> scores for triples
    # get_all_prediction_df() -> scores for triples

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


@dataclasses.dataclass
class TargetPredictions(Predictions):
    """Targets with their predicted scores."""

    # get_prediction_df(head | relation | tail) -> score targets

    target: Target
    other_columns_fixed_ids: Tuple[int, int]

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


@dataclasses.dataclass
class ScorePack:
    """A pair of result triples and scores."""

    #: the ID-based triples, shape: (n, 3)
    result: torch.LongTensor

    #: the scores
    scores: torch.FloatTensor

    def process(self, factory: Optional[CoreTriplesFactory] = None, **kwargs) -> "TriplePredictions":
        """Start post-processing scores."""
        if factory is None:
            df = tensor_to_df(self.result, score=self.scores, **kwargs)
        else:
            df = factory.tensor_to_df(self.result, score=self.scores, **kwargs)
        return TriplePredictions(df=df, factory=factory)


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


def isin_many_dim(elements: torch.Tensor, test_elements: torch.Tensor, dim: int = 0) -> torch.BoolTensor:
    """Return whether elements are contained in test elements."""
    inverse, counts = torch.cat([elements, test_elements], dim=dim).unique(
        return_counts=True, return_inverse=True, dim=dim
    )[1:]
    return counts[inverse[: elements.shape[dim]]] > 1


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

    pred = TargetPredictions(df=rv, factory=triples_factory, target=target, other_columns_fixed_ids=other_col_ids)
    if remove_known:
        pred = pred.filter_triples(triples_factory, testing)
    if add_novelties:
        pred = pred.add_membership_columns(training=triples_factory, testing=testing)
    return pred.df


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


@torch.inference_mode()
def predict(
    model: Model,
    *,
    k: Optional[int] = None,
    batch_size: Optional[int] = 1,
    mode: Optional[InductiveMode] = None,
    target: Target = LABEL_TAIL,
) -> ScorePack:
    """Calculate and store scores for either all triples, or the top k triples.

    :param model:
        A PyKEEN model
    :param k:
        The number of triples to return. Set to ``None`` to keep all.
    :param batch_size:
        The batch size to use for calculating scores; set to `None` to determine largest possible batch size
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.
    :param target:
        the prediction target to use. Prefer targets which are efficient to predict with the given model,
        e.g., tails for ConvE.

    :return:
        A score pack of parallel triples and scores
    """
    # set model to evaluation mode
    model.eval()

    logger.warning(
        f"predict is an expensive operation, involving {model.num_entities ** 2 * model.num_real_relations:,} "
        f"score evaluations.",
    )

    consumer: ScoreConsumer
    if k is None:
        logger.warning(
            "Not providing k to `predict` entails huge memory requirements for reasonably-sized knowledge graphs.",
        )
        consumer = AllScoreConsumer(num_entities=model.num_entities, num_relations=model.num_relations)
    else:
        consumer = TopKScoreConsumer(k=k, device=model.device)
    dataset = AllPredictionDataset(
        num_entities=model.num_entities, num_relations=model.num_real_relations, target=target
    )
    consume_scores(model, dataset, consumer, batch_size=batch_size or len(dataset), mode=mode)
    return consumer.finalize()


# note type alias annotation required,
# cf. https://mypy.readthedocs.io/en/stable/common_issues.html#variables-vs-type-aliases
# batch, TODO: ids?
PredictionBatch: TypeAlias = torch.LongTensor


class ScoreConsumer:
    """A consumer of scores for visitor pattern."""

    result: torch.LongTensor
    scores: torch.FloatTensor
    flatten: bool

    @abstractmethod
    def __call__(
        self,
        batch: PredictionBatch,
        target: Target,
        scores: torch.FloatTensor,
    ) -> None:
        """Consume scores for the given hr_batch."""
        raise NotImplementedError

    def finalize(self) -> ScorePack:
        """Finalize the result to build a score pack."""
        return _build_pack(result=self.result, scores=self.scores, flatten=self.flatten)


class CountScoreConsumer(ScoreConsumer):
    """A simple consumer which counts the number of batches and scores."""

    def __init__(self) -> None:
        """Initialize the consumer."""
        super().__init__()
        self.batch_count = 0
        self.score_count = 0

    # docstr-coverage: inherited
    def __call__(
        self,
        batch: PredictionBatch,
        target: Target,
        scores: torch.FloatTensor,
    ) -> None:  # noqa: D102
        self.batch_count += batch.shape[0]
        self.score_count += scores.numel()


class TopKScoreConsumer(ScoreConsumer):
    """Collect top-k triples & scores."""

    flatten = False

    def __init__(self, k: int = 3, device: DeviceHint = None) -> None:
        """
        Initialize the consumer.

        :param k:
            the number of top-scored triples to collect
        :param device:
            the model's device
        """
        self.k = k
        device = resolve_device(device=device)
        # initialize buffer on device
        self.result = torch.empty(0, 3, dtype=torch.long, device=device)
        self.scores = torch.empty(0, device=device)

    # docstr-coverage: inherited
    def __call__(
        self,
        batch: PredictionBatch,
        target: Target,
        scores: torch.FloatTensor,
    ) -> None:  # noqa: D102
        batch_size, num_scores = scores.shape
        assert batch.shape == (batch_size, 2)

        # reshape, shape: (batch_size * num_entities,)
        top_scores = scores.view(-1)

        # get top scores within batch
        if top_scores.numel() >= self.k:
            top_scores, top_indices = top_scores.topk(
                k=min(self.k, top_scores.numel()),
                largest=True,
                sorted=False,
            )
            # determine corresponding indices
            # batch_id, score_id = divmod(top_indices, num_scores)
            batch_id = torch.div(top_indices, num_scores, rounding_mode="trunc")
            score_id = top_indices % num_scores
            key_indices = batch[batch_id]
        else:
            key_indices = batch.unsqueeze(dim=1).repeat(1, num_scores, 1).view(-1, 2)
            score_id = torch.arange(num_scores, device=batch.device).view(1, -1).repeat(batch_size, 1).view(-1)

        # combine to top triples
        j = 0
        triples = []
        for col in COLUMN_LABELS:
            if col == target:
                index = score_id
            else:
                index = key_indices[:, j]
                j += 1
            triples.append(index)
        top_triples = torch.stack(triples, dim=-1)

        # append to global top scores
        self.scores = torch.cat([self.scores, top_scores])
        self.result = torch.cat([self.result, top_triples])

        # reduce size if necessary
        if self.result.shape[0] > self.k:
            self.scores, indices = self.scores.topk(k=self.k, largest=True, sorted=False)
            self.result = self.result[indices]


class AllScoreConsumer(ScoreConsumer):
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
        self.scores = torch.empty(num_entities, num_relations, num_entities, device="cpu")
        # Explicitly create triples
        self.result = torch.stack(
            [
                torch.arange(num_entities).view(-1, 1, 1).repeat(1, num_relations, num_entities),
                torch.arange(num_relations).view(1, -1, 1).repeat(num_entities, 1, num_entities),
                torch.arange(num_entities).view(1, 1, -1).repeat(num_entities, num_relations, 1),
            ],
            dim=-1,
        ).view(-1, 3)

    # docstr-coverage: inherited
    def __call__(
        self,
        batch: PredictionBatch,
        target: Target,
        scores: torch.FloatTensor,
    ) -> None:  # noqa: D102
        j = 0
        selectors: List[Union[slice, torch.LongTensor]] = []
        for col in COLUMN_LABELS:
            if col == target:
                selector = slice(None)
            else:
                selector = batch[:, j]
                j += 1
            selectors.append(selector)
        if target == LABEL_HEAD:
            scores = scores.t()
        self.scores[selectors[0], selectors[1], selectors[2]] = scores.to(self.scores.device)


class PredictionDataset(torch.utils.data.Dataset):
    """A base class for prediction datasets."""

    def __init__(self, target: Target = LABEL_TAIL) -> None:
        """Initialize the dataset.

        :param target:
            the prediction target to use. Prefer targets which are efficient to predict with the given model,
            e.g., tails for ConvE.
        """
        super().__init__()
        # TODO: variable targets across batches/samples?
        self.target = target

    @abstractmethod
    def __getitem__(self, item: int) -> PredictionBatch:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class AllPredictionDataset(PredictionDataset):
    """A dataset for predicting all possible triples."""

    def __init__(self, num_entities: int, num_relations: int, **kwargs) -> None:
        """Initialize the dataset.

        :param num_entities:
            the number of entities
        :param num_relations:
            the number of relations
        :param kwargs:
            additional keyword-based parameters passed to :meth:`PredictionDataset.__init__`
        """
        super().__init__(**kwargs)
        self.num_entities = num_entities
        self.num_relations = num_relations
        # (?, r, t) => r.stride > t.stride
        # (h, ?, t) => h.stride > t.stride
        # (h, r, ?) => h.stride > r.stride
        self.divisor = num_relations if self.target == LABEL_TAIL else num_entities

    # docstr-coverage: inherited
    def __len__(self) -> int:  # noqa: D102
        if self.target == LABEL_RELATION:
            return self.num_entities**2
        return self.num_entities * self.num_relations

    # docstr-coverage: inherited
    def __getitem__(self, item: int) -> torch.LongTensor:  # noqa: D102
        quotient, remainder = divmod(item, self.divisor)
        return torch.as_tensor([quotient, remainder])


Restriction = Union[torch.LongTensor, Collection[int], int]


class PartiallyRestrictedPredictionDataset(PredictionDataset):
    r"""
    A dataset for scoring some links.

    "Some links" is defined as

    .. math ::
        \mathcal{T}_{interest} = \mathcal{E}_{h} \times \mathcal{R}_{r} \times \mathcal{E}_{t}

    .. note ::
        For now, the target, i.e., position whose prediction method in the model is utilized,
        must be the full set of entities/relations.

    Example
    .. code-block:: python

        # train model; note: needs larger number of epochs to do something useful ;-)
        from pykeen.pipeline import pipeline
        result = pipeline(dataset="nations", model="mure", training_kwargs=dict(num_epochs=0))

        # create prediction dataset, where the head entities is from a set of European countries,
        # and the relations are connected to tourism
        from pykeen.models.predict import PartiallyRestrictedPredictionDataset
        heads = result.training.entities_to_ids(entities=["netherlands", "poland", "uk"])
        relations = result.training.relations_to_ids(relations=["reltourism", "tourism", "tourism3"])
        dataset = PartiallyRestrictedPredictionDataset(heads=heads, relations=relations)

        # calculate all scores for this restricted set, and keep k=3 largest
        from pykeen.models.predict import consume_scores, TopKScoreConsumer
        consumer = TopKScoreConsumer(k=3)
        consume_scores(result.model, ds, consumer)
        score_pack = consumer.finalize()

        # add labels
        df = result.training.tensor_to_df(score_pack.result, score=score_pack.scores)
    """

    #: the choices for the first and second component of the input batch
    parts: Tuple[torch.LongTensor, torch.LongTensor]

    def __init__(
        self,
        *,
        heads: Optional[Restriction] = None,
        relations: Optional[Restriction] = None,
        tails: Optional[Restriction] = None,
        target: Target = LABEL_TAIL,
    ) -> None:
        """
        Initialize restricted prediction dataset.

        :param heads:
            the restricted head entities
        :param relations:
            the restricted relations
        :param tails:
            the restricted tails
        :param target:
            the prediction target

        :raises NotImplementedError:
            if the target position restricted, or any non-target position is not restricted
        """
        super().__init__(target=target)
        parts: List[torch.LongTensor] = []
        for restriction, on in zip((heads, relations, tails), COLUMN_LABELS):
            if on == target:
                if restriction is not None:
                    raise NotImplementedError("Restrictions on the target are not yet supported.")
                continue
            if restriction is None:
                raise NotImplementedError("Requires size info")
            elif isinstance(restriction, int):
                restriction = [restriction]
            restriction = torch.as_tensor(restriction)
            parts.append(restriction)
        assert len(parts) == 2
        self.parts = (parts[0], parts[1])  # for mypy

    # docstr-coverage: inherited
    def __len__(self) -> int:  # noqa: D102
        return math.prod(map(len, self.parts))

    # docstr-coverage: inherited
    def __getitem__(self, item: int) -> PredictionBatch:  # noqa: D102
        quotient, remainder = divmod(item, len(self.parts[0]))
        return torch.as_tensor([self.parts[0][quotient], self.parts[1][remainder]])


@torch.inference_mode()
@maximize_memory_utilization(parameter_name="batch_size", keys=["model", "dataset", "consumers", "mode"])
def consume_scores(
    model: Model,
    dataset: PredictionDataset,
    *consumers: ScoreConsumer,
    batch_size: int = 1,
    mode: Optional[InductiveMode] = None,
) -> None:
    """
    Batch-wise calculation of all triple scores and consumption.

    From a high-level perspective, this method does the following:

    .. code-block:: python

        for batch in dataset:
            scores = model.predict(batch)
            for consumer in consumers:
                consumer(batch, scores)

    By bringing custom prediction datasets and/or score consumers, this method is highly configurable.

    :param model:
        the model used to calculate scores
    :param dataset:
        the dataset defining the prediction tasks, i.e., inputs to `model.predict` to loop over.
    :param consumers:
        the consumers of score batches
    :param batch_size:
        the batch size to use. Will automatically be lowered, if the hardware cannot handle this large batch sizes
    :param mode:
        The pass mode, which is None in the transductive setting and one of "training",
        "validation", or "testing" in the inductive setting.

    :raises ValueError:
        if no score consumers are given
    """
    if not consumers:
        raise ValueError("Did not receive any consumer")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    for batch in tqdm(data_loader, desc="scoring", unit="batch", unit_scale=True, leave=False):
        batch = batch.to(model.device)
        # calculate batch scores onces
        scores = model.predict(batch, target=dataset.target, full_batch=False, mode=mode)
        # consume by all consumers
        for consumer in consumers:
            consumer(batch, target=dataset.target, scores=scores)


def _build_pack(result: torch.LongTensor, scores: torch.FloatTensor, flatten: bool = False) -> ScorePack:
    """Sort final result and package in a score pack."""
    scores, indices = torch.sort(scores.flatten() if flatten else scores, descending=True)
    result = result[indices]
    return ScorePack(result=result, scores=scores)


@maximize_memory_utilization(keys=["model"])
def _predict_triples_batched(
    model: Model,
    mapped_triples: MappedTriples,
    batch_size: int,
    *,
    mode: Optional[InductiveMode],
) -> torch.FloatTensor:
    """Predict scores for triples in batches."""
    return torch.cat(
        [
            model.predict_hrt(hrt_batch=hrt_batch, mode=mode)
            for hrt_batch in mapped_triples.split(split_size=batch_size, dim=0)
        ],
        dim=0,
    )


@torch.inference_mode()
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
    # normalize input
    triples = get_mapped_triples(triples, factory=triples_factory)
    # calculate scores (with automatic memory optimization)
    scores = _predict_triples_batched(
        model=model, mapped_triples=triples, batch_size=batch_size or len(triples), mode=mode
    ).squeeze(dim=1)

    if triples_factory is None:
        return tensor_to_df(tensor=triples, score=scores)

    return triples_factory.tensor_to_df(tensor=triples, score=scores)
