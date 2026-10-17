"""Mock models that return fixed scores.

These are useful for baselines.
"""

from collections.abc import Mapping
from typing import Any, ClassVar

import torch

from .base import Model
from .scoring import ScoringBatch, TargetScoringBatch, TripleScoringBatch
from ..triples.triples_factory import KGInfo
from ..typing import LABEL_RELATION, FloatTensor, InductiveMode

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

    def __init__(self, *, triples_factory: KGInfo, **kwargs) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the (training) triples factory
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Model.__init__`
        """
        super().__init__(triples_factory=triples_factory, **kwargs)
        self.num_entities = triples_factory.num_entities
        self.num_relations = triples_factory.num_relations

        # This empty 1-element tensor doesn't actually do anything,
        # but is necessary since models with no grad params blow
        # up the optimizer
        self.dummy = torch.nn.Parameter(torch.empty(1), requires_grad=True)

    def collect_regularization_term(self):  # noqa: D102
        return 0.0

    def _get_entity_len(self, mode: InductiveMode | None) -> int:
        if mode is not None:
            raise NotImplementedError
        return self.num_entities

    def _reset_parameters_(self):
        pass  # Not needed for mock model

    def _generate_fake_scores(
        self,
        h: FloatTensor,
        r: FloatTensor,
        t: FloatTensor,
    ) -> FloatTensor:
        """Generate fake scores."""
        return (h * (self.num_entities * self.num_relations) + r * self.num_entities + t).float().requires_grad_(True)

    def _score(self, batch: ScoringBatch, *, mode: InductiveMode | None = None) -> FloatTensor:
        match batch:
            case TripleScoringBatch():
                return self._generate_fake_scores(*batch.indices)
            case TargetScoringBatch():
                h, r, t = batch.lookup_indices
                target_ids = batch.target_ids
                if target_ids is None:
                    num = self.num_relations if batch.target == LABEL_RELATION else self.num_entities
                    target_ids = torch.arange(num, device=batch.device)
                match batch.target:
                    case "head":
                        h = target_ids
                    case "relation":
                        r = target_ids
                    case "tail":
                        t = target_ids
                return self._generate_fake_scores(h, r, t)
