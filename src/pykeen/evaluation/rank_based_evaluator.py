# -*- coding: utf-8 -*-

"""Implementation of ranked based evaluator."""

import itertools
import logging
import math
import random
from collections import defaultdict
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union, cast

import numpy as np
import pandas as pd
import torch
from class_resolver import HintOrType, OptionalKwargs

from .evaluator import Evaluator, MetricResults, prepare_filter_triples
from .ranking_metric_lookup import MetricKey
from .ranks import Ranks
from ..metrics.ranking import RankBasedMetric, rank_based_metric_resolver
from ..metrics.utils import Metric
from ..triples.triples_factory import CoreTriplesFactory
from ..typing import (
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    RANK_TYPES,
    SIDE_BOTH,
    SIDES,
    ExtendedTarget,
    MappedTriples,
    RankType,
    Target,
)

__all__ = [
    "RankBasedEvaluator",
    "RankBasedMetricResults",
]

logger = logging.getLogger(__name__)

RANKING_METRICS: Mapping[str, Type[Metric]] = {cls().key: cls for cls in rank_based_metric_resolver}

K = TypeVar("K")


def _flatten(nested: Mapping[K, Sequence[np.ndarray]]) -> Mapping[K, np.ndarray]:
    return {key: np.concatenate(value) for key, value in nested.items()}


def _iter_ranks(
    ranks: Mapping[Tuple[Target, RankType], Sequence[np.ndarray]],
    num_candidates: Mapping[Target, Sequence[np.ndarray]],
) -> Iterable[Tuple[ExtendedTarget, RankType, np.ndarray, np.ndarray]]:
    sides = sorted(num_candidates.keys())
    # flatten dictionaries
    ranks_flat = _flatten(ranks)
    num_candidates_flat = _flatten(num_candidates)
    for rank_type in RANK_TYPES:
        # individual side
        for side in sides:
            yield side, rank_type, ranks_flat[side, rank_type], num_candidates_flat[side]

        # combined
        c_ranks = np.concatenate([ranks_flat[side, rank_type] for side in sides])
        c_num_candidates = np.concatenate([num_candidates_flat[side] for side in sides])
        yield SIDE_BOTH, rank_type, c_ranks, c_num_candidates


class RankBasedMetricResults(MetricResults):
    """Results from computing metrics."""

    data: MutableMapping[Tuple[str, ExtendedTarget, RankType], float]

    metrics = RANKING_METRICS

    @classmethod
    def from_dict(cls, **kwargs):
        """Create an instance from kwargs."""
        return cls(kwargs)

    @classmethod
    def from_ranks(
        cls,
        metrics: Iterable[RankBasedMetric],
        rank_and_candidates: Iterable[Tuple[ExtendedTarget, RankType, np.ndarray, np.ndarray]],
    ) -> "RankBasedMetricResults":
        """Create rank-based metric results from the given rank/candidate sets."""
        return cls(
            data={
                (metric.key, target, rank_type): metric(ranks=ranks, num_candidates=num_candidates)
                for metric, (target, rank_type, ranks, num_candidates) in itertools.product(
                    metrics, rank_and_candidates
                )
            }
        )

    @classmethod
    def create_random(cls, random_state: Union[None, int, random.Random] = None) -> "RankBasedMetricResults":
        """Create random results useful for testing."""
        if random_state is None or isinstance(random_state, int):
            random_state = random.Random(random_state)
        data = {}
        for metric_cls in rank_based_metric_resolver:
            metric = metric_cls()
            low, high = metric_cls.value_range.lower, metric_cls.value_range.upper
            if high is None or math.isinf(high):
                high = 1000.0
            for target, rank_type in itertools.product(SIDES, RANK_TYPES):
                data[metric.key, target, rank_type] = random_state.uniform(low, high)
        return cls(data=data)

    def get_metric(self, name: str) -> float:
        """Get the rank-based metric.

        :param name: The name of the metric, created by concatenating three parts:

            1. The side (one of "head", "tail", or "both"). Most publications exclusively report "both".
            2. The type (one of "optimistic", "pessimistic", "realistic")
            3. The metric name ("adjusted_mean_rank_index", "adjusted_mean_rank", "mean_rank, "mean_reciprocal_rank",
               "inverse_geometric_mean_rank",
               or "hits@k" where k defaults to 10 but can be substituted for an integer. By default, 1, 3, 5, and 10
               are available. Other K's can be calculated by setting the appropriate variable in the
               ``evaluation_kwargs`` in the :func:`pykeen.pipeline.pipeline` or setting ``ks`` in the
               :class:`pykeen.evaluation.RankBasedEvaluator`.

            In general, all metrics are available for all combinations of sides/types except AMR and AMRI, which
            are only calculated for the average type. This is because the calculation of the expected MR in the
            optimistic and pessimistic case scenarios is still an active area of research and therefore has no
            implementation yet.
        :return: The value for the metric
        :raises ValueError: if an invalid name is given.

        Get the average MR

        >>> metric_results.get('both.realistic.mean_rank')

        If you only give a metric name, it assumes that it's for "both" sides and "realistic" type.

        >>> metric_results.get('adjusted_mean_rank_index')

        This function will do its best to infer what's going on if you only specify one part.

        >>> metric_results.get('left.mean_rank')
        >>> metric_results.get('optimistic.mean_rank')

        Get the default Hits @ K (where $k=10$)

        >>> metric_results.get('hits@k')

        Get a given Hits @ K

        >>> metric_results.get('hits@5')
        """
        return self._get_metric(MetricKey.lookup(name))

    def _get_metric(self, metric_key: MetricKey) -> float:
        for (metric_key_, target, rank_type), value in self.data.items():
            if MetricKey(metric=metric_key_, side=target, rank_type=rank_type) == metric_key:
                return value
        raise KeyError(metric_key)

    def to_dict(self) -> Mapping[ExtendedTarget, Mapping[RankType, Mapping[str, float]]]:  # noqa: D102
        result: MutableMapping[ExtendedTarget, MutableMapping[RankType, MutableMapping[str, float]]] = {}
        for side, rank_type, metric_name, metric_value in self._iter_rows():
            result.setdefault(side, {})
            result[side].setdefault(rank_type, {})
            result[side][rank_type][metric_name] = metric_value
        return result

    def to_flat_dict(self):  # noqa: D102
        return {f"{side}.{rank_type}.{metric_name}": value for side, rank_type, metric_name, value in self._iter_rows()}

    def to_df(self) -> pd.DataFrame:
        """Output the metrics as a pandas dataframe."""
        return pd.DataFrame(list(self._iter_rows()), columns=["Side", "Type", "Metric", "Value"])

    def _iter_rows(self) -> Iterable[Tuple[ExtendedTarget, RankType, str, Union[float, int]]]:
        for (metric_key, side, rank_type), value in self.data.items():
            yield side, rank_type, metric_key, value


class RankBasedEvaluator(Evaluator):
    """A rank-based evaluator for KGE models."""

    num_entities: Optional[int]
    ranks: MutableMapping[Tuple[Target, RankType], List[np.ndarray]]
    num_candidates: MutableMapping[Target, List[np.ndarray]]

    def __init__(
        self,
        filtered: bool = True,
        metrics: Optional[Sequence[HintOrType[RankBasedMetric]]] = None,
        metrics_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """Initialize rank-based evaluator.

        :param filtered:
            Whether to use the filtered evaluation protocol. If enabled, ranking another true triple higher than the
            currently considered one will not decrease the score.
        :param metrics:
            the rank-based metrics to compute
        :param metrics_kwargs:
            additional keyword parameter
        :param kwargs: Additional keyword arguments that are passed to the base class.
        """
        super().__init__(
            filtered=filtered,
            requires_positive_mask=False,
            **kwargs,
        )
        if metrics is None:
            assert metrics_kwargs is None
            metrics = [key for key in rank_based_metric_resolver.options if key != "hits_at_k"]
            metrics_kwargs = [None] * len(metrics)
            metrics += ["hits_at_k"] * 4
            metrics_kwargs += [dict(k=k) for k in (1, 3, 5, 10)]

        self.metrics = rank_based_metric_resolver.make_many(metrics, metrics_kwargs)
        self.ranks = defaultdict(list)
        self.num_candidates = defaultdict(list)
        self.num_entities = None

    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: torch.FloatTensor,
        true_scores: Optional[torch.FloatTensor] = None,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        if true_scores is None:
            raise ValueError(f"{self.__class__.__name__} needs the true scores!")

        batch_ranks = Ranks.from_scores(
            true_score=true_scores,
            all_scores=scores,
        )
        self.num_entities = scores.shape[1]
        for rank_type, v in batch_ranks.items():
            self.ranks[target, rank_type].append(v.detach().cpu().numpy())
        self.num_candidates[target].append(batch_ranks.number_of_options.detach().cpu().numpy())

    def finalize(self) -> RankBasedMetricResults:  # noqa: D102
        if self.num_entities is None:
            raise ValueError
        result = RankBasedMetricResults.from_ranks(
            metrics=self.metrics,
            rank_and_candidates=_iter_ranks(ranks=self.ranks, num_candidates=self.num_candidates),
        )
        # Clear buffers
        self.ranks.clear()
        self.num_candidates.clear()

        return result


def sample_negatives(
    evaluation_triples: MappedTriples,
    additional_filter_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
    num_samples: int = 50,
    num_entities: Optional[int] = None,
) -> Mapping[Target, torch.FloatTensor]:
    """
    Sample true negatives for sampled evaluation.

    :param evaluation_triples: shape: (n, 3)
        the evaluation triples
    :param additional_filter_triples:
        additional true triples which are to be filtered
    :param num_samples: >0
        the number of samples
    :param num_entities:
        the number of entities

    :return:
        A mapping of sides to negative samples
    """
    additional_filter_triples = prepare_filter_triples(
        mapped_triples=evaluation_triples,
        additional_filter_triples=additional_filter_triples,
    )
    num_entities = num_entities or (additional_filter_triples[:, [0, 2]].max().item() + 1)
    columns = [LABEL_HEAD, LABEL_RELATION, LABEL_TAIL]
    num_triples = evaluation_triples.shape[0]
    df = pd.DataFrame(data=evaluation_triples.numpy(), columns=columns)
    all_df = pd.DataFrame(data=additional_filter_triples.numpy(), columns=columns)
    id_df = df.reset_index()
    all_ids = set(range(num_entities))
    negatives = {}
    for side in [LABEL_HEAD, LABEL_TAIL]:
        this_negatives = cast(torch.FloatTensor, torch.empty(size=(num_triples, num_samples), dtype=torch.long))
        other = [c for c in columns if c != side]
        for _, group in pd.merge(id_df, all_df, on=other, suffixes=["_eval", "_all"]).groupby(
            by=other,
        ):
            pool = list(all_ids.difference(group[f"{side}_all"].unique().tolist()))
            if len(pool) < num_samples:
                logger.warning(
                    f"There are less than num_samples={num_samples} candidates for side={side}, triples={group}.",
                )
                # repeat
                pool = int(math.ceil(num_samples / len(pool))) * pool
            for i in group["index"].unique():
                this_negatives[i, :] = torch.as_tensor(
                    data=random.sample(population=pool, k=num_samples),
                    dtype=torch.long,
                )
        negatives[side] = this_negatives
    return negatives


class SampledRankBasedEvaluator(RankBasedEvaluator):
    """
    A rank-based evaluator using sampled negatives instead of all negatives, cf. [teru2020]_.

    Notice that this evaluator yields optimistic estimations of the metrics evaluated on all entities,
    cf. https://arxiv.org/abs/2106.06935.
    """

    negatives: Mapping[Target, torch.LongTensor]

    def __init__(
        self,
        evaluation_factory: CoreTriplesFactory,
        *,
        additional_filter_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
        num_negatives: Optional[int] = None,
        head_negatives: Optional[torch.LongTensor] = None,
        tail_negatives: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Initialize the evaluator.

        :param evaluation_factory:
            the factory with evaluation triples
        :param head_negatives: shape: (num_triples, num_negatives)
            the entity IDs of negative samples for head prediction for each evaluation triple
        :param tail_negatives: shape: (num_triples, num_negatives)
            the entity IDs of negative samples for tail prediction for each evaluation triple
        :param kwargs:
            additional keyword-based arguments passed to RankBasedEvaluator.__init__
        """
        super().__init__(**kwargs)
        if head_negatives is None and tail_negatives is None:
            # default for inductive LP by [teru2020]
            num_negatives = num_negatives or 50
            logger.info(
                f"Sampling {num_negatives} negatives for each of the "
                f"{evaluation_factory.num_triples} evaluation triples.",
            )
            if num_negatives > evaluation_factory.num_entities:
                raise ValueError("Cannot use more negative samples than there are entities.")
            negatives = sample_negatives(
                evaluation_triples=evaluation_factory.mapped_triples,
                additional_filter_triples=additional_filter_triples,
                num_entities=evaluation_factory.num_entities,
                num_samples=num_negatives,
            )
        elif head_negatives is None or tail_negatives is None:
            raise ValueError("Either both, head and tail negatives must be provided, or none.")
        else:
            negatives = {
                LABEL_HEAD: head_negatives,
                LABEL_TAIL: tail_negatives,
            }

        # verify input
        for side, side_negatives in negatives.items():
            if side_negatives.shape[0] != evaluation_factory.num_triples:
                raise ValueError(f"Negatives for {side} are in wrong shape: {side_negatives.shape}")
        self.triple_to_index = {(h, r, t): i for i, (h, r, t) in enumerate(evaluation_factory.mapped_triples.tolist())}
        self.negative_samples = negatives
        self.num_entities = evaluation_factory.num_entities

    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: torch.FloatTensor,
        true_scores: Optional[torch.FloatTensor] = None,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        if true_scores is None:
            raise ValueError(f"{self.__class__.__name__} needs the true scores!")

        num_entities = scores.shape[1]
        # TODO: do not require to compute all scores beforehand
        triple_indices = [self.triple_to_index[h, r, t] for h, r, t in hrt_batch.cpu().tolist()]
        negative_entity_ids = self.negative_samples[target][triple_indices]
        negative_scores = scores[
            torch.arange(hrt_batch.shape[0], device=hrt_batch.device).unsqueeze(dim=-1),
            negative_entity_ids,
        ]
        # super.evaluation assumes that the true scores are part of all_scores
        scores = torch.cat([true_scores, negative_scores], dim=-1)
        super().process_scores_(
            hrt_batch=hrt_batch,
            target=target,
            scores=scores,
            true_scores=true_scores,
            dense_positive_mask=dense_positive_mask,
        )
        # write back correct num_entities
        # TODO: should we give num_entities in the constructor instead of inferring it every time ranks are processed?
        self.num_entities = num_entities
