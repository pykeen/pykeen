# -*- coding: utf-8 -*-

"""Utility class for storing ranks."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple, Union

import torch

from ..typing import RANK_OPTIMISTIC, RANK_PESSIMISTIC, RANK_REALISTIC, RankType

__all__ = [
    "Ranks",
    "RankBuilder",
]


@dataclass
class Ranks:
    """Ranks for different ranking types."""

    #: The optimistic rank is the rank when assuming all options with an equal score are placed
    #: behind the current test triple.
    #: shape: (batch_size,)
    optimistic: torch.FloatTensor

    #: The realistic rank is the average of the optimistic and pessimistic rank, and hence the expected rank
    #: over all permutations of the elements with the same score as the currently considered option.
    #: shape: (batch_size,)
    realistic: torch.FloatTensor

    #: The pessimistic rank is the rank when assuming all options with an equal score are placed
    #: in front of current test triple.
    #: shape: (batch_size,)
    pessimistic: torch.FloatTensor

    #: The number of options is the number of items considered in the ranking. It may change for
    #: filtered evaluation
    #: shape: (batch_size,)
    number_of_options: torch.LongTensor

    def items(self) -> Iterable[Tuple[RankType, torch.FloatTensor]]:
        """Iterate over pairs of rank types and their associated tensors."""
        yield from self.to_type_dict().items()

    def to_type_dict(self) -> Mapping[RankType, torch.FloatTensor]:
        """Return mapping from rank-type to rank value tensor."""
        return {
            RANK_OPTIMISTIC: self.optimistic,
            RANK_REALISTIC: self.realistic,
            RANK_PESSIMISTIC: self.pessimistic,
        }

    @classmethod
    def from_scores(
        cls,
        true_score: torch.FloatTensor,
        all_scores: torch.FloatTensor,
    ) -> "Ranks":
        """Compute ranks given scores.

        :param true_score: torch.Tensor, shape: (batch_size, 1)
            The score of the true triple.
        :param all_scores: torch.Tensor, shape: (batch_size, num_entities)
            The scores of all corrupted triples (including the true triple).

        :return:
            a data structure containing the (filtered) ranks.
        """
        # The optimistic rank is the rank when assuming all options with an
        # equal score are placed behind the currently considered. Hence, the
        # rank is the number of options with better scores, plus one, as the
        # rank is one-based.
        optimistic_rank = (all_scores > true_score).sum(dim=1) + 1

        # The pessimistic rank is the rank when assuming all options with an
        # equal score are placed in front of the currently considered. Hence,
        # the rank is the number of options which have at least the same score
        # minus one (as the currently considered option in included in all
        # options). As the rank is one-based, we have to add 1, which nullifies
        # the "minus 1" from before.
        pessimistic_rank = (all_scores >= true_score).sum(dim=1)

        # The realistic rank is the average of the optimistic and pessimistic
        # rank, and hence the expected rank over all permutations of the elements
        # with the same score as the currently considered option.
        realistic_rank = (optimistic_rank + pessimistic_rank).float() * 0.5

        # We set values which should be ignored to NaN, hence the number of options
        # which should be considered is given by
        number_of_options = torch.isfinite(all_scores).sum(dim=1)

        return cls(
            optimistic=optimistic_rank,
            pessimistic=pessimistic_rank,
            realistic=realistic_rank,
            number_of_options=number_of_options,
        )


@dataclass
class RankBuilder:
    """
    Incremental rank builder.

    The incremental rank builder can be used to calculate ranks whenever we do not have access to all candidate scores
    at once, but are able to iterate over (batches of) them, e.g., in combination with score slicing. And advantage
    over the current slicing implementation is that we do not need to be able to store all scores, but can already
    start aggregating them.

    In general, the following pattern can be used.

    We start by initializing the builder with a (batch of) true score(s)

    >>> import torch
    >>> from pykeen.evaluation.ranks import RankBuilder
    >>> builder = RankBuilder(y_true=torch.as_tensor([0.2, 0.1]))

    Now, we iterate over batches of scores, e.g., `2 * 15` scores at once. Note that all but the last dimension have
    to match the shape of `y_true`. Also notice that the operation does *not* work in-place, but rather returns an
    updated object. This does not pose a serious memory problem, since each RankBuilder object only holds references to
    tensors, and does not copy the tensors themselves.

    >>> for _ in range(10):
    ...    builder = builder.update(y_pred=torch.rand(2, 15))

    After we aggregated all batches, we can obtain the ranks by

    >>> ranks = builder.compute()

    Note that this pattern is somewhat similar to the one found in :mod:`torchmetrics`, except that it works on
    "sub-batch" / "slice" basis rather than batch level.
    """

    #: the scores of the true choice, shape: (*bs), dtype: float
    y_true: torch.Tensor

    #: the number of scores which were larger than the true score, shape: (*bs), dtype: long
    larger: Union[torch.Tensor, int] = 0

    #: the number of scores which were not smaller than the true score, shape: (*bs), dtype: long
    not_smaller: Union[torch.Tensor, int] = 0

    #: the total number of compared scores, shape: (*bs), dtype: long
    total: Union[torch.Tensor, int] = 0

    def update(self, y_pred: torch.FloatTensor) -> "RankBuilder":
        """Update the rank builder with a batch of scores.

        :param y_pred: shape: (*bs, partial_num_choices)
            the predicted scores, which are aligned to the batch of true scores

        :return:
            the updated rank builder
        """
        return RankBuilder(
            y_true=self.y_true,
            larger=self.larger + (y_pred > self.y_true.unsqueeze(dim=-1)).sum(dim=-1),
            not_smaller=self.not_smaller + (y_pred >= self.y_true.unsqueeze(dim=-1)).sum(dim=-1),
            total=self.total + torch.isfinite(y_pred).sum(dim=-1),
        )

    def compute(self) -> torch.Tensor:
        """Calculate the ranks for the aggregated counts.

        :return:
            a rank object for different rank types
        """
        optimistic = self.larger + 1
        pessimistic = self.not_smaller + 1
        realistic = optimistic + pessimistic
        if isinstance(realistic, torch.Tensor):
            realistic = realistic.float()
        realistic = 0.5 * realistic
        return Ranks(optimistic=optimistic, pessimistic=pessimistic, realistic=realistic, number_of_options=self.total)
