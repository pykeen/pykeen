# -*- coding: utf-8 -*-

"""Utility class for storing ranks."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

import torch

from ..typing import RANK_OPTIMISTIC, RANK_PESSIMISTIC, RANK_REALISTIC, RankType

__all__ = [
    "Ranks",
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
