# -*- coding: utf-8 -*-

"""Mock models that return fixed scores.

These are useful for baselines.
"""

from typing import Any, ClassVar, Mapping, Optional

import torch

from .base import Model
from ..triples.triples_factory import KGInfo
from ..typing import InductiveMode

__all__ = [
    "FixedModel",
]


class FixedModel(Model):
    r"""A mock model returning fixed scores.

    .. math ::
        score(h, r, t) = h \cdot |\mathcal{E}| \cdot |\mathcal{R}| + r \cdot |\mathcal{E}| + t

    ---
    name: Fixed Model
    citation:
        author: Berrendorf
        year: 2021
        link: https://github.com/pykeen/pykeen/pull/691
        github: pykeen/pykeen
    """

    hpo_default: ClassVar[Mapping[str, Any]] = {}

    def __init__(self, *, triples_factory: KGInfo, **_kwargs):
        """
        Initialize the model.

        :param triples_factory:
            the (training) triples factory
        :param _kwargs:
            ignored keyword-based parameters
        """
        super().__init__(
            triples_factory=triples_factory,
        )
        self.num_entities = triples_factory.num_entities
        self.num_relations = triples_factory.num_relations

        # This empty 1-element tensor doesn't actually do anything,
        # but is necessary since models with no grad params blow
        # up the optimizer
        self.dummy = torch.nn.Parameter(torch.empty(1), requires_grad=True)

    # docstr-coverage: inherited
    def collect_regularization_term(self):  # noqa: D102
        return 0.0

    def _get_entity_len(self, mode: Optional[InductiveMode]) -> int:
        if mode is not None:
            raise NotImplementedError
        return self.num_entities

    # docstr-coverage: inherited
    def _reset_parameters_(self):  # noqa: D102
        pass  # Not needed for mock model

    def _generate_fake_scores(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Generate fake scores."""
        return (h * (self.num_entities * self.num_relations) + r * self.num_entities + t).float().requires_grad_(True)

    # docstr-coverage: inherited
    def score_hrt(self, hrt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(*hrt_batch.t()).unsqueeze(dim=-1)

    # docstr-coverage: inherited
    def score_t(
        self, hr_batch: torch.LongTensor, tails: Optional[torch.LongTensor] = None, **kwargs
    ) -> torch.FloatTensor:  # noqa: D102
        if tails is None:
            tails = torch.arange(self.num_entities, device=hr_batch.device).unsqueeze(dim=0)
        return self._generate_fake_scores(h=hr_batch[:, 0:1], r=hr_batch[:, 1:2], t=tails)

    # docstr-coverage: inherited
    def score_r(
        self, ht_batch: torch.LongTensor, relations: Optional[torch.LongTensor] = None, **kwargs
    ) -> torch.FloatTensor:  # noqa: D102
        if relations is None:
            relations = torch.arange(self.num_relations, device=ht_batch.device).unsqueeze(dim=0)
        return self._generate_fake_scores(h=ht_batch[:, 0:1], r=relations, t=ht_batch[:, 1:2])

    # docstr-coverage: inherited
    def score_h(
        self, rt_batch: torch.LongTensor, heads: Optional[torch.LongTensor] = None, **kwargs
    ) -> torch.FloatTensor:  # noqa: D102
        if heads is None:
            heads = torch.arange(self.num_entities, device=rt_batch.device).unsqueeze(dim=0)
        return self._generate_fake_scores(h=heads, r=rt_batch[:, 0:1], t=rt_batch[:, 1:2])
