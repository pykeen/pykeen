# -*- coding: utf-8 -*-

"""Implementation of ranked based evaluator."""

import itertools as itt
import logging
import math
import random
import re
from collections import defaultdict
from dataclasses import fields
from typing import (
    Any,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)
from dataclasses import dataclass, field, fields
from typing import DefaultDict, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from class_resolver import Resolver, normalize_string
from typing_extensions import Literal

from .evaluator import Evaluator, MetricResults, prepare_filter_triples
from .metrics import HitsAtK, InverseArithmeticMeanRank, RankBasedMetric
from ..constants import SIDES
from ..triples.triples_factory import CoreTriplesFactory
from ..typing import (
    LABEL_HEAD,
    LABEL_TAIL,
    RANK_REALISTIC,
    RANK_TYPE_SYNONYMS,
    RANK_TYPES,
    MappedTriples,
    Ranks,
    RankType,
    Target,
)
from ..typing import LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, MappedTriples, Target
from ..utils import fix_dataclass_init_docs

__all__ = [
    "compute_rank_from_scores",
    "RankBasedEvaluator",
    "RankBasedMetricResults",
    "MetricKey",
    "resolve_metric_name",
    "metric_resolver",
]

logger = logging.getLogger(__name__)

# typing
ExtendedSide = Union[Target, Literal["both"]]
SIDE_BOTH: ExtendedSide = "both"
EXTENDED_SIDES: Tuple[ExtendedSide, ...] = cast(Tuple[ExtendedSide, ...], SIDES) + (SIDE_BOTH,)

metric_resolver: Resolver[RankBasedMetric] = Resolver.from_subclasses(
    base=RankBasedMetric,
    default=InverseArithmeticMeanRank,  # mrr
)
# also add synonyms
for cls in metric_resolver.lookup_dict.values():
    metric_resolver.register(cls=cls, synonyms=cls.synonyms, raise_on_conflict=False)


class MetricKey(NamedTuple):
    """A key for the kind of metric to resolve."""

    metric: Type[RankBasedMetric]
    side: ExtendedSide
    rank_type: RankType
    k: Optional[int]

    def __str__(self) -> str:  # noqa: D105
        components: List[Any] = [self.metric, self.side, self.rank_type]
        if self.k:
            components.append(self.k)
        return ".".join(map(str, components))

    @classmethod
    def resolve_metric_name(cls, name: str) -> "MetricKey":
        """Functional metric name normalization."""
        match = METRIC_PATTERN.match(normalize_string(name, suffix=None))
        if not match:
            raise ValueError(f"Invalid metric name: {name}")
        side: Union[str, ExtendedSide]
        rank_type: Union[str, RankType]
        kf: Union[None, str, int]
        kb: Union[None, str, int]
        name, side, rank_type, kf, kb = [match.group(key) for key in ("name", "side", "type", "kf", "kb")]
        k = kf or kb

        # normalize metric name
        if not name:
            raise ValueError("A metric name must be provided.")
        metric_cls = metric_resolver.lookup(name)

        # special case for hits_at_k
        if metric_cls is HitsAtK and k is not None:
            # TODO: Fractional?
            try:
                k = int(k)
            except ValueError as error:
                raise ValueError(f"Invalid k={k} for hits_at_k") from error
            if k < 0:
                raise ValueError(f"For hits_at_k, you must provide a positive value of k, but found {k}.")
        assert k is None or isinstance(k, int)

        # normalize side
        side = side or SIDE_BOTH
        side = side.lower()
        if side not in EXTENDED_SIDES:
            raise ValueError(f"Invalid side: {side}. Allowed are {EXTENDED_SIDES}.")

        # normalize rank type
        rank_type = rank_type or RANK_REALISTIC
        rank_type = rank_type.lower()
        rank_type = RANK_TYPE_SYNONYMS.get(rank_type, rank_type)
        if rank_type not in RANK_TYPES:
            raise ValueError(f"Invalid rank type: {rank_type}. Allowed are {RANK_TYPES}.")
        if rank_type not in metric_cls.supported_rank_types:
            raise ValueError(
                f"Invalid rank type for {metric_resolver.normalize_cls(metric_cls)}: {rank_type}. "
                f"Allowed type: {metric_cls.supported_rank_types}",
            )
        return MetricKey(metric_cls, side, rank_type, k)


def compute_rank_from_scores(
    true_score: torch.FloatTensor,
    all_scores: torch.FloatTensor,
) -> Ranks:
    """Compute ranks given scores.

    :param true_score: torch.Tensor, shape: (batch_size, 1)
        The score of the true triple.
    :param all_scores: torch.Tensor, shape: (batch_size, num_entities)
        The scores of all corrupted triples (including the true triple).

    :return:
        a data structure containing the (filtered) ranks.
    """
    # The optimistic rank is the rank when assuming all options with an equal score are placed behind the currently
    # considered. Hence, the rank is the number of options with better scores, plus one, as the rank is one-based.
    optimistic_rank = (all_scores > true_score).sum(dim=1) + 1

    # The pessimistic rank is the rank when assuming all options with an equal score are placed in front of the
    # currently considered. Hence, the rank is the number of options which have at least the same score minus one
    # (as the currently considered option in included in all options). As the rank is one-based, we have to add 1,
    # which nullifies the "minus 1" from before.
    pessimistic_rank = (all_scores >= true_score).sum(dim=1)

    # The realistic rank is the average of the optimistic and pessimistic rank, and hence the expected rank over
    # all permutations of the elements with the same score as the currently considered option.
    realistic_rank = (optimistic_rank + pessimistic_rank).float() * 0.5

    # We set values which should be ignored to NaN, hence the number of options which should be considered is given by
    number_of_options = torch.isfinite(all_scores).sum(dim=1)

    return Ranks(
        optimistic=optimistic_rank,
        realistic=realistic_rank,
        pessimistic=pessimistic_rank,
        number_of_options=number_of_options,
    )


_SIDE_PATTERN = "|".join(EXTENDED_SIDES)
_TYPE_PATTERN = "|".join(itt.chain(RANK_TYPES, RANK_TYPE_SYNONYMS.keys()))
# HITS_PATTERN = re.compile(r"(hits_at_|hits@|h@)(?P<kf>\d+)")
_METRIC_PATTERN = "|".join(itt.chain(metric_resolver.lookup_dict.keys(), metric_resolver.synonyms.keys()))
METRIC_PATTERN = re.compile(
    rf"^(?P<name>{_METRIC_PATTERN})(?P<kf>\d+)?(\.(?P<side>{_SIDE_PATTERN}))?(\.(?P<type>{_TYPE_PATTERN}))?(\.(?P<kb>\d+))?$",
)


# TODO: special hits@k


def resolve_metric_name(name: str) -> MetricKey:
    """Functional metric name normalization."""
    return MetricKey.resolve_metric_name(name)


class RankBasedMetricResults(MetricResults):
    """Results from computing metrics."""

    def __init__(self, results: Mapping[Tuple[str, ExtendedSide, RankType], float]):
        """Initialize the results."""
        self.results = results

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
        metric, side, rank_type, k = MetricKey.resolve_metric_name(name)
        if metric is not HitsAtK:
            return self.results[metric_resolver.normalize(metric), side, rank_type]
        raise NotImplementedError

    def to_flat_dict(self):  # noqa: D102
        return {f"{side}.{rank_type}.{metric_name}": value for side, rank_type, metric_name, value in self._iter_rows()}

    def to_df(self) -> pd.DataFrame:
        """Output the metrics as a pandas dataframe."""
        return pd.DataFrame(list(self._iter_rows()), columns=["Side", "Type", "Metric", "Value"])

    def _iter_rows(self) -> Iterable[Tuple[ExtendedSide, RankType, str, Union[float, int]]]:
        for side, rank_type in itt.product(EXTENDED_SIDES, RANK_TYPES):
            for k, v in self.hits_at_k[side][rank_type].items():
                yield side, rank_type, f"hits_at_{k}", v
            for f in fields(self):
                if f.name == "hits_at_k":
                    continue
                side_data = getattr(self, f.name)[side]
                if rank_type in side_data:
                    yield side, rank_type, f.name, side_data[rank_type]


class RankBasedEvaluator(Evaluator):
    """A rank-based evaluator for KGE models."""

    #: the actual rank data
    ranks: MutableMapping[RankType, MutableMapping[Target, List[np.ndarray]]]

    #: the number of choices for each ranking task; relevant for expected metrics
    number_of_options: MutableMapping[Target, List[np.ndarray]]

    #: the rank-based metrics to compute
    metrics: Mapping[str, RankBasedMetric]

    def __init__(
        self,
        ks: Optional[Iterable[Union[int, float]]] = None,
        filtered: bool = True,
        **kwargs,
    ):
        r"""Initialize rank-based evaluator.

        :param ks:
            The values for which to calculate hits@k. Defaults to $\{1, 3, 5, 10\}$.
        :param filtered:
            Whether to use the filtered evaluation protocol. If enabled, ranking another true triple higher than the
            currently considered one will not decrease the score.
        :param kwargs: Additional keyword arguments that are passed to the base class.
        """
        super().__init__(
            filtered=filtered,
            requires_positive_mask=False,
            **kwargs,
        )
        ks = tuple(ks) if ks is not None else (1, 3, 5, 10)
        for k in ks:
            if isinstance(k, float) and not (0 < k < 1):
                raise ValueError(
                    "If k is a float, it should represent a relative rank, i.e. a value between 0 and 1 (excl.)",
                )
        metrics = [
            metric_resolver.make(query=query) for query in metric_resolver.lookup_dict.values() if query is not HitsAtK
        ] + [metric_resolver.make(HitsAtK, k=k) for k in ks]
        self.metrics = {metric_resolver.normalize_inst(metric): metric for metric in metrics}
        self.ranks = {rank_type: {side: [] for side in SIDES} for rank_type in RANK_TYPES}
        self.number_of_options = defaultdict(list)

    def _update_ranks_(
        self,
        true_scores: torch.FloatTensor,
        all_scores: torch.FloatTensor,
        side: Target,
        hrt_batch: MappedTriples,
    ) -> None:
        """Shared code for updating the stored ranks for head/tail scores.

        :param true_scores: shape: (batch_size,)
        :param all_scores: shape: (batch_size, num_entities)
        """
        batch_ranks = compute_rank_from_scores(
            true_score=true_scores,
            all_scores=all_scores,
        )
        for rank_type, ranks in batch_ranks.to_type_dict().items():
            self.ranks[rank_type][side].append(ranks.detach().cpu().numpy())
        self.number_of_options[side].append(batch_ranks.number_of_options.detach().cpu().numpy())

    def process_tail_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        self._update_ranks_(true_scores=true_scores, all_scores=scores, side=LABEL_TAIL, hrt_batch=hrt_batch)

    def process_head_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        self._update_ranks_(true_scores=true_scores, all_scores=scores, side=LABEL_HEAD, hrt_batch=hrt_batch)

    @classmethod
    def _get_for_side(
        cls,
        mapping: Mapping[Target, List[np.ndarray]],
        side: ExtendedSide,
    ) -> np.ndarray:
        values: List[np.ndarray]
        if side == SIDE_BOTH:
            return np.concatenate([cls._get_for_side(mapping=mapping, side=_side) for _side in SIDES])
        else:
            return np.concatenate(mapping.get(cast(Target, side), [])).astype(dtype=np.float64)

    def finalize(self) -> RankBasedMetricResults:  # noqa: D102
        result: MutableMapping[Tuple[str, ExtendedSide, RankType], float] = dict()

        for side in EXTENDED_SIDES:
            num_candidates = self._get_for_side(mapping=self.number_of_options, side=side)
            if len(num_candidates) < 1:
                logger.warning(f"No num_candidates for side={side}")
                continue
            for rank_type in RANK_TYPES:
                ranks = self._get_for_side(mapping=self.ranks[rank_type], side=side)
                if len(ranks) < 1:
                    logger.warning(f"No ranks for side={side}, rank_type={rank_type}")
                    continue
                for metric_name, metric in self.metrics.items():
                    if rank_type not in metric.supported_rank_types:
                        continue
                    result[(metric_name, side, rank_type)] = metric(ranks=ranks, num_candidates=num_candidates)
        # Clear buffers
        self.ranks.clear()
        self.number_of_options.clear()
        return RankBasedMetricResults(result)


def sample_negatives(
    evaluation_triples: MappedTriples,
    side: Target,
    additional_filter_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
    num_samples: int = 50,
    num_entities: Optional[int] = None,
) -> torch.LongTensor:
    """
    Sample true negatives for sampled evaluation.

    :param evaluation_triples: shape: (n, 3)
        the evaluation triples
    :param side:
        the side for which to generate negatives
    :param additional_filter_triples:
        additional true triples which are to be filtered
    :param num_samples: >0
        the number of samples
    :param num_entities:
        the maximum Id for the given side

    :return: shape: (n, num_negatives)
        the negatives for the selected side prediction
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
    negatives = []
    for side in [LABEL_HEAD, LABEL_TAIL]:
        this_negatives = torch.empty(size=(num_triples, num_samples), dtype=torch.long)
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
        negatives.append(this_negatives)
    return negatives[0], negatives[1]


class SampledRankBasedEvaluator(RankBasedEvaluator):
    """
    A rank-based evaluator using sampled negatives instead of all negatives, cf. [teru2020]_.

    Notice that this evaluator yields optimistic estimations of the metrics evaluated on all entities,
    cf. https://arxiv.org/abs/2106.06935.
    """

    #: the negative samples for each side
    negative_samples: Mapping[Target, torch.LongTensor]

    def __init__(
        self,
        evaluation_factory: CoreTriplesFactory,
        *,
        additional_filter_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
        num_negatives: Optional[int] = None,
        negatives: Optional[Mapping[Target, Optional[torch.LongTensor]]] = None,
        **kwargs,
    ):
        """
        Initialize the evaluator.

        :param evaluation_factory:
            the factory with evaluation triples
        :param negatives: shape: (num_triples, num_negatives)
            the entity IDs of negative samples for head/tail prediction for each evaluation triple
        :param kwargs:
            additional keyword-based arguments passed to RankBasedEvaluator.__init__
        """
        super().__init__(**kwargs)
        if negatives is None:
            negatives = {side: None for side in SIDES}
        # make sure that negatives is mutable
        negatives = dict(negatives)
        for side in negatives.keys():
            # default for inductive LP by [teru2020]
            if negatives[side] is not None:
                continue
            logger.info(
                f"Sampling {num_negatives} negatives for each of the "
                f"{evaluation_factory.num_triples} evaluation triples.",
            )
            num_negatives = num_negatives or 50
            if num_negatives > evaluation_factory.num_entities:
                raise ValueError("Cannot use more negative samples than there are entities.")
            negatives[side] = sample_negatives(
                evaluation_triples=evaluation_factory.mapped_triples,
                side=side,
                additional_filter_triples=additional_filter_triples,
                max_id=evaluation_factory.num_entities,
                num_samples=num_negatives,
            )

        # verify input
        for side, side_negatives in negatives.items():
            assert side_negatives is not None
            if side_negatives.shape[0] != evaluation_factory.num_triples:
                raise ValueError(f"Negatives for side={side} are in wrong shape: {side_negatives.shape}")
        self.triple_to_index = {(h, r, t): i for i, (h, r, t) in enumerate(evaluation_factory.mapped_triples.tolist())}
        self.negative_samples = negatives
        self.num_entities = evaluation_factory.num_entities

    def _update_ranks_(
        self,
        true_scores: torch.FloatTensor,
        all_scores: torch.FloatTensor,
        side: Target,
        hrt_batch: MappedTriples,
    ) -> None:  # noqa: D102
        # TODO: do not require to compute all scores beforehand
        triple_indices = [self.triple_to_index[h, r, t] for h, r, t in hrt_batch.cpu().tolist()]
        negative_entity_ids = self.negative_samples[side][triple_indices]
        negative_scores = all_scores[
            torch.arange(hrt_batch.shape[0], device=hrt_batch.device).unsqueeze(dim=-1),
            negative_entity_ids,
        ]
        # super.evaluation assumes that the true scores are part of all_scores
        scores = torch.cat([true_scores, negative_scores], dim=-1)
        super()._update_ranks_(true_scores=true_scores, all_scores=scores, side=side, hrt_batch=hrt_batch)
        # write back correct num_entities
        # TODO: should we give num_entities in the constructor instead of inferring it every time ranks are processed?
        self.num_entities = all_scores.shape[1]


def numeric_expected_value(
    metric: str,
    num_candidates: Union[Sequence[int], np.ndarray],
    num_samples: int,
) -> float:
    """
    Compute expected metric value by summation.

    Depending on the metric, the estimate may not be very accurate and converage slowly, cf.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.expect.html
    """
    metric_func = metric_resolver.make(metric)
    num_candidates = np.asarray(num_candidates)
    generator = np.random.default_rng()
    expectation = 0
    for _ in range(num_samples):
        ranks = generator.integers(low=0, high=num_candidates)
        expectation += metric_func(ranks)
    return expectation / num_samples
