# -*- coding: utf-8 -*-

"""Implementation of ranked based evaluator."""

import itertools
import logging
import math
import random
from collections import defaultdict
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union, cast

import numpy as np
import numpy.random
import pandas
import pandas as pd
import torch
from class_resolver import HintOrType, OptionalKwargs

from .evaluator import Evaluator, MetricResults, prepare_filter_triples
from .ranking_metric_lookup import MetricKey
from .ranks import Ranks
from ..constants import TARGET_TO_INDEX
from ..metrics.ranking import HITS_METRICS, RankBasedMetric, rank_based_metric_resolver
from ..metrics.utils import Metric
from ..triples.triples_factory import CoreTriplesFactory
from ..typing import (
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    RANK_OPTIMISTIC,
    RANK_PESSIMISTIC,
    RANK_REALISTIC,
    RANK_TYPES,
    SIDE_BOTH,
    ExtendedTarget,
    MappedTriples,
    RankType,
    Target,
)

__all__ = [
    "RankBasedEvaluator",
    "RankBasedMetricResults",
    "sample_negatives",
    "SampledRankBasedEvaluator",
    "MacroRankBasedEvaluator",
]

logger = logging.getLogger(__name__)

RANKING_METRICS: Mapping[str, Type[Metric]] = {cls().key: cls for cls in rank_based_metric_resolver}

K = TypeVar("K")


def _flatten(nested: Mapping[K, Sequence[np.ndarray]]) -> Mapping[K, np.ndarray]:
    return {key: np.concatenate(value) for key, value in nested.items()}


def _iter_ranks(
    ranks: Mapping[Tuple[Target, RankType], Sequence[np.ndarray]],
    num_candidates: Mapping[Target, Sequence[np.ndarray]],
    weights: Optional[Mapping[Target, Sequence[np.ndarray]]] = None,
) -> Iterable[Tuple[ExtendedTarget, RankType, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    # terminate early if there are no ranks
    if not ranks:
        logger.debug("Empty ranks. This should only happen during size probing.")
        return

    sides = sorted(num_candidates.keys())
    # flatten dictionaries
    ranks_flat = _flatten(ranks)
    num_candidates_flat = _flatten(num_candidates)
    weights_flat: Mapping[Target, np.ndarray]
    if weights is None:
        weights_flat = dict()
    else:
        weights_flat = _flatten(weights)
    for rank_type in RANK_TYPES:
        # individual side
        for side in sides:
            yield side, rank_type, ranks_flat[side, rank_type], num_candidates_flat[side], weights_flat.get(side)

        # combined
        c_ranks = np.concatenate([ranks_flat[side, rank_type] for side in sides])
        c_num_candidates = np.concatenate([num_candidates_flat[side] for side in sides])
        c_weights = None if weights is None else np.concatenate([weights_flat[side] for side in sides])
        yield SIDE_BOTH, rank_type, c_ranks, c_num_candidates, c_weights


class RankBasedMetricResults(MetricResults):
    """Results from computing metrics."""

    data: MutableMapping[Tuple[str, ExtendedTarget, RankType], float]

    metrics = RANKING_METRICS

    @classmethod
    def from_ranks(
        cls,
        metrics: Iterable[RankBasedMetric],
        rank_and_candidates: Iterable[Tuple[ExtendedTarget, RankType, np.ndarray, np.ndarray, Optional[np.ndarray]]],
    ) -> "RankBasedMetricResults":
        """Create rank-based metric results from the given rank/candidate sets."""
        return cls(
            data={
                (metric.key, target, rank_type): metric(ranks=ranks, num_candidates=num_candidates, weights=weights)
                for metric, (target, rank_type, ranks, num_candidates, weights) in itertools.product(
                    metrics, rank_and_candidates
                )
            }
        )

    @classmethod
    def create_random(cls, random_state: Optional[int] = None) -> "RankBasedMetricResults":
        """Create random results useful for testing."""
        generator = numpy.random.default_rng(seed=random_state)
        num_candidates = generator.integers(low=2, high=1000, size=(2, 1000))
        ranks = generator.integers(low=1, high=num_candidates[None], size=(2, 2, 1000))
        ranks = numpy.maximum.accumulate(ranks, axis=1)  # increasing, since order of RANK_TYPES
        data = {}
        target_to_idx = {
            LABEL_HEAD: 0,
            LABEL_TAIL: 1,
            SIDE_BOTH: [0, 1],
        }
        rank_to_idx = {
            RANK_OPTIMISTIC: [0],
            RANK_PESSIMISTIC: [1],
            RANK_REALISTIC: [0, 1],
        }
        for metric_cls in rank_based_metric_resolver:
            metric = metric_cls()
            for target, i in target_to_idx.items():
                for rank_type, j in rank_to_idx.items():
                    this_ranks = ranks[i, j].mean(axis=0).flatten()
                    data[metric.key, target, rank_type] = metric(ranks=this_ranks, num_candidates=num_candidates[i])
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

        :raises: ValueError
            if an invalid name is given.

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
        """
        Get the value of the metric corresponding to the given metric key.

        :param metric_key:
            the metric key.

        :return:
            the metric value.

        :raises KeyError:
            if no metric could be found matching the given key
        """
        for (metric_key_, target, rank_type), value in self.data.items():
            if MetricKey(metric=metric_key_, side=target, rank_type=rank_type) == metric_key:
                return value
        raise KeyError(metric_key)

    # docstr-coverage: inherited
    def to_dict(self) -> Mapping[ExtendedTarget, Mapping[RankType, Mapping[str, float]]]:  # noqa: D102
        result: MutableMapping[ExtendedTarget, MutableMapping[RankType, MutableMapping[str, float]]] = {}
        for side, rank_type, metric_name, metric_value in self._iter_rows():
            result.setdefault(side, {})
            result[side].setdefault(rank_type, {})
            result[side][rank_type][metric_name] = metric_value
        return result

    # docstr-coverage: inherited
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
        add_defaults: bool = True,
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
        :param add_defaults:
            whether to add all default metrics besides the ones specified by `metrics` / `metrics_kwargs`.
        :param kwargs: Additional keyword arguments that are passed to the base class.
        """
        super().__init__(
            filtered=filtered,
            requires_positive_mask=False,
            **kwargs,
        )
        if metrics is None:
            add_defaults = True
            metrics = []
        self.metrics = rank_based_metric_resolver.make_many(metrics, metrics_kwargs)
        if add_defaults:
            hits_at_k_keys = [rank_based_metric_resolver.normalize_cls(cls) for cls in HITS_METRICS]
            ks = (1, 3, 5, 10)
            metrics = [key for key in rank_based_metric_resolver.lookup_dict if key not in hits_at_k_keys]
            metrics_kwargs = [None] * len(metrics)
            for hits_at_k_key in hits_at_k_keys:
                metrics += [hits_at_k_key] * len(ks)
                metrics_kwargs += [dict(k=k) for k in ks]
            self.metrics.extend(rank_based_metric_resolver.make_many(metrics, metrics_kwargs))
        self.ranks = defaultdict(list)
        self.num_candidates = defaultdict(list)
        self.num_entities = None

    # docstr-coverage: inherited
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

    # docstr-coverage: inherited
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
    """A rank-based evaluator using sampled negatives instead of all negatives.

    See also [teru2020]_.

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
        :param additional_filter_triples:
            additional true triples to use for filtering; only relevant if not explicit negatives are given.
            cf. :func:`pykeen.evaluation.rank_based_evaluator.sample_negatives`
        :param num_negatives:
            the number of negatives to sample; only relevant if not explicit negatives are given.
            cf. :func:`pykeen.evaluation.rank_based_evaluator.sample_negatives`
        :param head_negatives: shape: (num_triples, num_negatives)
            the entity IDs of negative samples for head prediction for each evaluation triple
        :param tail_negatives: shape: (num_triples, num_negatives)
            the entity IDs of negative samples for tail prediction for each evaluation triple
        :param kwargs:
            additional keyword-based arguments passed to
            :meth:`pykeen.evaluation.rank_based_evaluator.RankBasedEvaluator.__init__`

        :raises ValueError:
            if only a single side's negatives are given, or the negatives are in wrong shape
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

    # docstr-coverage: inherited
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


class MacroRankBasedEvaluator(RankBasedEvaluator):
    """Macro-average rank-based evaluation."""

    COLUMNS = (LABEL_HEAD, LABEL_RELATION, LABEL_TAIL)
    precomputed_weights: Mapping[Target, Mapping[Tuple[int, int], float]]
    weights: MutableMapping[Target, List[numpy.ndarray]]

    def __init__(
        self,
        *,
        evaluation_factory: Optional[CoreTriplesFactory] = None,
        evaluation_triples: Optional[MappedTriples] = None,
        **kwargs,
    ):
        """
        Initialize the evaluator.

        :param evaluation_factory:
            the evaluation triples' factory. Must be provided, if no explicit triples are provided.
        :param evaluation_triples:
            the evaluation triples. If given, takes precedence over extracting triples from a factory.
        :param kwargs:
            additional keyword-based parameters passed to :meth:`RankBasedEvaluator.__init__`.

        :raises ValueError:
            if neither evaluation triples nor a factory are provided
        """
        super().__init__(**kwargs)
        if evaluation_triples is None:
            if evaluation_factory is None:
                raise ValueError("Need to provide either evaluation_triples or evaluation_factory.")
            evaluation_triples = evaluation_factory.mapped_triples
        # compute macro weights
        df = pandas.DataFrame(data=evaluation_triples.numpy(), columns=list(self.COLUMNS))
        self.precomputed_weights = dict()
        self.weights = {}
        for target in (LABEL_HEAD, LABEL_TAIL):
            key = self._get_key(target)
            counts = df.groupby(by=key).nunique()[target]
            key_list = cast(Iterable[Tuple[int, int]], map(tuple, counts.index.tolist()))
            self.precomputed_weights[target] = dict(
                zip(key_list, numpy.reciprocal(counts.values.astype(float)).tolist())
            )
            self.weights[target] = []

    def _get_key(self, target: Target) -> List[Target]:
        return [c for c in self.COLUMNS if c != target]

    # docstr-coverage: inherited
    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: torch.FloatTensor,
        true_scores: Optional[torch.FloatTensor] = None,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        super().process_scores_(
            hrt_batch=hrt_batch,
            target=target,
            scores=scores,
            true_scores=true_scores,
            dense_positive_mask=dense_positive_mask,
        )
        key_list = (
            hrt_batch[:, [TARGET_TO_INDEX[key] for key in self._get_key(target=target)]].detach().numpy().tolist()
        )
        keys = cast(List[Tuple[int, int]], list(map(tuple, key_list)))
        self.weights[target].append(numpy.asarray([self.precomputed_weights[target][k] for k in keys]))

    # docstr-coverage: inherited
    def finalize(self) -> RankBasedMetricResults:  # noqa: D102
        if self.num_entities is None:
            raise ValueError
        result = RankBasedMetricResults.from_ranks(
            metrics=self.metrics,
            rank_and_candidates=_iter_ranks(ranks=self.ranks, num_candidates=self.num_candidates, weights=self.weights),
        )
        # Clear buffers
        self.weights.clear()
        self.ranks.clear()
        self.num_candidates.clear()

        return result
