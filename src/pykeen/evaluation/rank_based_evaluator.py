"""Implementation of ranked based evaluator."""

from __future__ import annotations

import functools
import itertools
import logging
import math
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from typing import (
    NamedTuple,
    TypeVar,
)

import numpy as np
import numpy.random
import pandas as pd
import torch
from class_resolver import OneOrManyHintOrType, OneOrManyOptionalKwargs

from .evaluator import Evaluator, MetricResults, prepare_filter_triples
from .ranks import Ranks
from ..constants import COLUMN_LABELS, TARGET_TO_KEY_LABELS, TARGET_TO_KEYS
from ..metrics.ranking import HITS_METRICS, RankBasedMetric, rank_based_metric_resolver
from ..metrics.utils import Metric
from ..triples.triples_factory import CoreTriplesFactory
from ..typing import (
    LABEL_HEAD,
    LABEL_TAIL,
    RANK_REALISTIC,
    RANK_TYPE_SYNONYMS,
    RANK_TYPES,
    SIDE_BOTH,
    SIDES,
    ExtendedTarget,
    FloatTensor,
    LongTensor,
    MappedTriples,
    RankType,
    Target,
    normalize_rank_type,
    normalize_target,
)

__all__ = [
    "RankBasedEvaluator",
    "RankBasedMetricResults",
    "sample_negatives",
    "SampledRankBasedEvaluator",
    "MacroRankBasedEvaluator",
]

logger = logging.getLogger(__name__)

RANKING_METRICS: Mapping[str, type[Metric]] = {cls().key: cls for cls in rank_based_metric_resolver}

K = TypeVar("K")


def _flatten(nested: Mapping[K, Sequence[np.ndarray]]) -> Mapping[K, np.ndarray]:
    return {key: np.concatenate(value) for key, value in nested.items()}


class RankPack(NamedTuple):
    """A pack of ranks for aggregation."""

    target: ExtendedTarget
    rank_type: RankType
    ranks: np.ndarray
    num_candidates: np.ndarray
    weights: np.ndarray | None

    def resample(self, seed: int | None) -> RankPack:
        """Resample rank pack."""
        generator = np.random.default_rng(seed=seed)
        n = len(self.ranks)
        ids = generator.integers(n, size=(n,))
        weights = None if self.weights is None else self.weights[ids]
        return RankPack(
            target=self.target,
            rank_type=self.rank_type,
            ranks=self.ranks[ids],
            num_candidates=self.num_candidates[ids],
            weights=weights,
        )


def _iter_ranks(
    ranks: Mapping[tuple[Target, RankType], Sequence[np.ndarray]],
    num_candidates: Mapping[Target, Sequence[np.ndarray]],
    weights: Mapping[Target, Sequence[np.ndarray]] | None = None,
) -> Iterable[RankPack]:
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
            yield RankPack(
                side, rank_type, ranks_flat[side, rank_type], num_candidates_flat[side], weights_flat.get(side)
            )

        # combined
        c_ranks = np.concatenate([ranks_flat[side, rank_type] for side in sides])
        c_num_candidates = np.concatenate([num_candidates_flat[side] for side in sides])
        c_weights = None if weights is None else np.concatenate([weights_flat[side] for side in sides])
        yield RankPack(SIDE_BOTH, rank_type, c_ranks, c_num_candidates, c_weights)


class RankBasedMetricKey(NamedTuple):
    """A key for ranking-based metrics."""

    side: ExtendedTarget
    rank_type: RankType
    metric: str


# parsing metrics
# metric pattern = side?.type?.metric.k?
_SIDE_PATTERN = "|".join(SIDES)
_TYPE_PATTERN = "|".join(itertools.chain(RANK_TYPES, RANK_TYPE_SYNONYMS.keys()))
METRIC_PATTERN = re.compile(
    rf"^((?P<side>{_SIDE_PATTERN})\.)?((?P<type>{_TYPE_PATTERN})\.)?(?P<name>[\w@]+)(\.(?P<k>\d+))?$",
)
HITS_PATTERN = re.compile(r"(?P<name>h@|hits@|hits_at_)(?P<k>\d+)")


class RankBasedMetricResults(MetricResults[RankBasedMetricKey]):
    """Results from computing metrics."""

    metrics = RANKING_METRICS

    @classmethod
    def key_from_string(cls, s: str | None) -> RankBasedMetricKey:
        """Get the rank-based metric key.

        The key input is understood as a dot-separated composition of

        1. The side (one of "head", "tail", or "both"). Most publications exclusively report "both".
           If not given "both" is assumed.
        2. The rank type (one of "optimistic", "pessimistic", "realistic"). If not given, "realistic" is assumed.
        3. The metric name, e.g., "adjusted_mean_rank_index", "adjusted_mean_rank", "mean_rank, "mean_reciprocal_rank",
            "inverse_geometric_mean_rank", or "hits@k" where k defaults to 10 but can be substituted for an integer.
            By default, 1, 3, 5, and 10 are available. Other K's can be calculated by setting the appropriate
            variable in the ``evaluation_kwargs`` in the :func:`pykeen.pipeline.pipeline` or setting ``ks`` in the
            :class:`pykeen.evaluation.RankBasedEvaluator`.

        In general, all metrics are available for all combinations of sides/types except AMR and AMRI, which
        are only calculated for the average type. This is because the calculation of the expected MR in the
        optimistic and pessimistic case scenarios is still an active area of research and therefore has no
        implementation yet.

        :param s:
            a string denoting a metric key

        :return: The resolved key.

        :raises ValueError:
            if the string cannot be resolved to a metric key

        Get the average MR

        >>> RankBasedMetricResults.key_from_string('both.realistic.mean_rank')
        RankBasedMetricKey(side='both', rank_type='realistic', metric='arithmetic_mean_rank')

        If you only give a metric name, it assumes that it's for 'both' sides and 'realistic' type.

        >>> RankBasedMetricResults.key_from_string('adjusted_mean_rank_index')
        RankBasedMetricKey(side='both', rank_type='realistic', metric='adjusted_arithmetic_mean_rank_index')

        This function will do its best to infer what's going on if you only specify one part.

        >>> RankBasedMetricResults.key_from_string('head.mean_rank')
        RankBasedMetricKey(side='head', rank_type='realistic', metric='arithmetic_mean_rank')

        >>> RankBasedMetricResults.key_from_string('optimistic.mean_rank')
        RankBasedMetricKey(side='both', rank_type='optimistic', metric='arithmetic_mean_rank')

        Get the default Hits @ K (where $k=10$)

        >>> RankBasedMetricResults.key_from_string('hits@k')
        RankBasedMetricKey(side='both', rank_type='realistic', metric='hits_at_10')

        Get a given Hits @ K

        >>> RankBasedMetricResults.key_from_string('hits@5')
        RankBasedMetricKey(side='both', rank_type='realistic', metric='hits_at_5')
        """
        if s is None:
            return RankBasedMetricKey(
                side=SIDE_BOTH, rank_type=RANK_REALISTIC, metric=rank_based_metric_resolver.make(query=None).key
            )

        match = METRIC_PATTERN.match(s)
        if not match:
            raise ValueError(f"Invalid metric name: {s}")
        k: None | str | int
        name, side, rank_type, k = (match.group(key) for key in ("name", "side", "type", "k"))
        name = name.lower()
        match = HITS_PATTERN.match(name)
        if match:
            name, k = match.groups()

        # normalize metric name
        if not name:
            raise ValueError("A metric name must be provided.")
        metric_cls = rank_based_metric_resolver.lookup(name)

        kwargs = {}
        if issubclass(metric_cls, HITS_METRICS):
            k = int(k or 10)
            assert k > 0
            kwargs["k"] = k

        metric = rank_based_metric_resolver.make(metric_cls, kwargs)

        # normalize side
        side = normalize_target(side)

        # normalize rank type
        rank_type = normalize_rank_type(rank_type)
        if rank_type not in metric.supported_rank_types:
            raise ValueError(
                f"Invalid rank type for {metric}: {rank_type}. Allowed type: {metric.supported_rank_types}"
            )

        return RankBasedMetricKey(side=side, rank_type=rank_type, metric=metric.key)

    @classmethod
    def from_ranks(
        cls,
        metrics: Iterable[RankBasedMetric],
        rank_and_candidates: Iterable[RankPack],
    ) -> RankBasedMetricResults:
        """Create rank-based metric results from the given rank/candidate sets."""
        return cls(
            data={
                RankBasedMetricKey(side=pack.target, rank_type=pack.rank_type, metric=metric.key): metric(
                    ranks=pack.ranks, num_candidates=pack.num_candidates, weights=pack.weights
                )
                for metric, pack in itertools.product(metrics, rank_and_candidates)
            }
        )

    @classmethod
    def create_random(cls, random_state: int | None = None) -> RankBasedMetricResults:
        """Create random results useful for testing."""
        targets = [LABEL_HEAD, LABEL_TAIL]
        num_targets = len(targets)
        num_rank_types = len(RANK_TYPES)
        generator = numpy.random.default_rng(seed=random_state)
        num_candidates = generator.integers(low=2, high=1000, size=(num_targets, 1000))
        ranks = generator.integers(low=1, high=num_candidates[None], size=(num_rank_types - 1, num_targets, 1000))
        # ensure that rank-opt <= rank-pess
        ranks = numpy.sort(ranks, axis=0)
        # assert that rank-real = (opt + pess)/2
        ranks = numpy.stack([ranks[0], (ranks[0] + ranks[1]) / 2, ranks[1]], axis=0)
        data: dict[RankBasedMetricKey | str, float] = {}
        # fixme: the annotation of ClassResolver.__iter__ seems to be broken (X instead of Type[X])
        metric_cls: type[RankBasedMetric]
        for metric_cls in rank_based_metric_resolver:
            metric = metric_cls()
            for (target, i), (j, rank_type) in itertools.product(RANDOM_TARGET_SLICE, enumerate(RANK_TYPES)):
                this_ranks = ranks[j, i].flatten()
                data[RankBasedMetricKey(side=target, rank_type=rank_type, metric=metric.key)] = metric(
                    ranks=this_ranks, num_candidates=num_candidates[i].flatten()
                )
        return cls(data=data)


RANDOM_TARGET_SLICE: list[tuple[ExtendedTarget, int | slice]] = [
    (LABEL_HEAD, 0),
    (LABEL_TAIL, 1),
    (SIDE_BOTH, slice(None)),
]


class RankBasedEvaluator(Evaluator[RankBasedMetricKey]):
    """A rank-based evaluator for KGE models."""

    metric_result_cls = RankBasedMetricResults
    num_entities: int | None
    ranks: MutableMapping[tuple[Target, RankType], list[np.ndarray]]
    num_candidates: MutableMapping[Target, list[np.ndarray]]

    def __init__(
        self,
        filtered: bool = True,
        metrics: OneOrManyHintOrType = None,
        metrics_kwargs: OneOrManyOptionalKwargs = None,
        add_defaults: bool = True,
        clear_on_finalize: bool = True,
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
        :param clear_on_finalize:
            whether to clear buffers on `finalize` call

            .. warning ::
                disabling this option may lead to memory leaks and incorrect results when used from the pipeline

        :param kwargs:
            Additional keyword arguments that are passed to the base class.
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
        self.clear_on_finalize = clear_on_finalize

    # docstr-coverage: inherited
    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: FloatTensor,
        true_scores: FloatTensor | None = None,
        dense_positive_mask: FloatTensor | None = None,
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
    def clear(self) -> None:  # noqa: D102
        self.ranks.clear()
        self.num_candidates.clear()

    # docstr-coverage: inherited
    def finalize(self) -> RankBasedMetricResults:  # noqa: D102
        if self.num_entities is None:
            raise ValueError
        result = RankBasedMetricResults.from_ranks(
            metrics=self.metrics,
            rank_and_candidates=_iter_ranks(ranks=self.ranks, num_candidates=self.num_candidates),
        )
        if self.clear_on_finalize:
            self.clear()
        return result

    def finalize_multi(self, n_boot: int = 1_000, seed: int = 42) -> Mapping[str, Sequence[float]]:
        """Bootstrap from :meth:`finalize`.

        :param n_boot:
            the number of resampling steps
        :param seed:
            the random seed.

        :return:
            a flat dictionary from metric names to list of values
        """
        result: defaultdict[str, list[float]] = defaultdict(list)

        for i in range(n_boot):
            rank_and_candidates = _iter_ranks(ranks=self.ranks, num_candidates=self.num_candidates)
            rank_and_candidates = map(functools.partial(RankPack.resample, seed=seed + i), rank_and_candidates)
            single_result = RankBasedMetricResults.from_ranks(
                metrics=self.metrics, rank_and_candidates=rank_and_candidates
            )
            for k, v in single_result.to_flat_dict().items():
                result[k].append(v)
        return result

    def finalize_with_confidence(
        self,
        estimator: str | Callable[[Sequence[float]], float] = np.median,
        ci: int | str | Callable[[Sequence[float]], float] = 90,
        n_boot: int = 1_000,
        seed: int = 42,
    ) -> Mapping[str, tuple[float, float]]:
        """Finalize result with confidence estimation via bootstrapping.

        Start by training a model (here, only for a one epochs)

        >>> from pykeen.pipeline import pipeline
        >>> result = pipeline(dataset="nations", model="rotate", training_kwargs=dict(num_epochs=1))

        Create an evaluator with `clear_on_finalize` set to `False`, e.g., via

        >>> from pykeen.evaluation import evaluator_resolver
        >>> evaluator = evaluator_resolver.make("rankbased", clear_on_finalize=False)

        Evaluate *once*, this time ignoring the result

        >>> evaluator.evaluate(model=result.model, mapped_triples=result.training.mapped_triples)

        Now, call `finalize_with_confidence` to obtain estimates for metrics together with confidence intervals

        >>> evaluator.finalize_with_confidence(n_boot=10)

        :param estimator:
            the estimator of central tendency.
        :param ci:
            the confidence interval
        :param n_boot:
            the number of resamplings to use for bootstrapping
        :param seed:
            the random seed

        :return:
            a dictionary from metric names to (central tendency, confidence) pairs
        """
        return {
            k: summarize_values(vs, estimator=estimator, ci=ci)
            for k, vs in self.finalize_multi(n_boot=n_boot, seed=seed).items()
        }


def _resolve_estimator(estimator: str | Callable[[Sequence[float]], float]) -> Callable[[Sequence[float]], float]:
    if callable(estimator):
        return estimator
    return getattr(np, estimator)


def _resolve_confidence(ci: int | str | Callable[[Sequence[float]], float]) -> Callable[[Sequence[float]], float]:
    if callable(ci):
        return ci
    if isinstance(ci, int | float):
        if ci < 0 or ci > 100:
            raise ValueError(f"Invalid CI value: {ci}. Must be in [0, 100].")
        ci_half = ci / 2.0

        def ipr(vs: Sequence[float]) -> float:
            """Return the inter-percentile range."""
            return np.diff(np.percentile(vs, q=[ci_half, 100 - ci_half])).item()

        return ipr
    return getattr(np, ci)


def summarize_values(
    vs: Sequence[float],
    estimator: str | Callable[[Sequence[float]], float] = np.median,
    ci: int | str | Callable[[Sequence[float]], float] = 90,
) -> tuple[float, float]:
    """Summarize values.

    :param vs:
        the values
    :param estimator:
        the central tendency estimator
    :param ci:
        the confidence estimator

    :return:
        a tuple estimates of central tendency and confidence
    """
    estimator = _resolve_estimator(estimator=estimator)
    ci = _resolve_confidence(ci=ci)
    return estimator(vs), ci(vs)


def sample_negatives(
    evaluation_triples: MappedTriples,
    additional_filter_triples: None | MappedTriples | list[MappedTriples] = None,
    num_samples: int = 50,
    num_entities: int | None = None,
) -> Mapping[Target, FloatTensor]:
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
    num_triples = evaluation_triples.shape[0]
    df = pd.DataFrame(data=evaluation_triples.numpy(), columns=COLUMN_LABELS)
    all_df = pd.DataFrame(data=additional_filter_triples.numpy(), columns=COLUMN_LABELS)
    id_df = df.reset_index()
    all_ids = set(range(num_entities))
    negatives = {}
    for side in [LABEL_HEAD, LABEL_TAIL]:
        this_negatives = torch.empty(size=(num_triples, num_samples), dtype=torch.long)
        other = TARGET_TO_KEY_LABELS[side]
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

    negative_samples: Mapping[Target, LongTensor]

    def __init__(
        self,
        evaluation_factory: CoreTriplesFactory,
        *,
        additional_filter_triples: None | MappedTriples | list[MappedTriples] = None,
        num_negatives: int | None = None,
        head_negatives: LongTensor | None = None,
        tail_negatives: LongTensor | None = None,
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
            if additional_filter_triples is not None:
                logger.warning(f"Ignoring parameter additional_filter_triples={additional_filter_triples}")

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
        scores: FloatTensor,
        true_scores: FloatTensor | None = None,
        dense_positive_mask: FloatTensor | None = None,
    ) -> None:  # noqa: D102
        if true_scores is None:
            raise ValueError(f"{self.__class__.__name__} needs the true scores!")

        num_entities = scores.shape[1]
        # TODO: do not require to compute all scores beforehand
        # cf. Model.score_t(ts=...)
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

    weights: MutableMapping[Target, list[np.ndarray]]

    def __init__(self, **kwargs):
        """
        Initialize the evaluator.

        :param kwargs:
            additional keyword-based parameters passed to :meth:`RankBasedEvaluator.__init__`.
        """
        super().__init__(**kwargs)
        self.keys = defaultdict(list)

    @staticmethod
    def _calculate_weights(keys: Iterable[np.ndarray]) -> np.ndarray:
        """Calculate macro weights, i.e., weights inversely proportional to the key frequency.

        :param keys:
            the keys, in batches

        :return: shape: (n,)
            the weights
        """
        # combine key batches
        keys = np.concatenate(list(keys), axis=0)
        # calculate key frequency
        inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)[1:]  # type:ignore
        # weight = inverse frequency
        weights = np.reciprocal(counts, dtype=float)
        # broadcast to samples
        return weights[inverse]

    # docstr-coverage: inherited
    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: FloatTensor,
        true_scores: FloatTensor | None = None,
        dense_positive_mask: FloatTensor | None = None,
    ) -> None:  # noqa: D102
        super().process_scores_(
            hrt_batch=hrt_batch,
            target=target,
            scores=scores,
            true_scores=true_scores,
            dense_positive_mask=dense_positive_mask,
        )
        # store keys for calculating macro weights
        self.keys[target].append(hrt_batch[:, TARGET_TO_KEYS[target]].detach().cpu().numpy())

    # docstr-coverage: inherited
    def clear(self) -> None:  # noqa: D102
        super().clear()
        self.keys.clear()

    # docstr-coverage: inherited
    def finalize(self) -> RankBasedMetricResults:  # noqa: D102
        if self.num_entities is None:
            raise ValueError
        # compute macro weights
        # note: we wrap the array into a list to be able to re-use _iter_ranks
        weights = {target: [self._calculate_weights(keys=keys)] for target, keys in self.keys.items()}
        # calculate weighted metrics
        result = RankBasedMetricResults.from_ranks(
            metrics=self.metrics,
            rank_and_candidates=_iter_ranks(ranks=self.ranks, num_candidates=self.num_candidates, weights=weights),
        )
        # Clear buffers
        self.clear()

        return result
