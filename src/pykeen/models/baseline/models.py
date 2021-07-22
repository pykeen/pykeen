# -*- coding: utf-8 -*-

"""Non-parametric baseline models."""

from abc import ABC

import numpy
import torch

from .utils import get_csr_matrix, marginal_score
from ..base import Model
from ...triples import CoreTriplesFactory

__all__ = [
    'EvaluationOnlyModel',
    'MarginalDistributionBaseline',
]


class EvaluationOnlyModel(Model, ABC):
    """A model which only implements the methods used for evaluation."""

    def __init__(self, triples_factory: CoreTriplesFactory):
        """Non-parametric models take a minimal set of arguments.

        :param triples_factory: The training triples factory is used to assign the number of entities, relations,
            and inverse condition in the non-parametric model.
        """
        super().__init__(
            triples_factory=triples_factory,
            # These operations are deterministic and a random seed can be fixed
            # just to avoid warnings
            random_seed=0,
            # These operations do not need to be performed on a GPU
            preferred_device='cpu',
        )

    def _reset_parameters_(self):
        """Non-parametric models do not implement :meth:`Model._reset_parameters_`."""
        raise RuntimeError

    def collect_regularization_term(self):  # noqa:D102
        """Non-parametric models do not implement :meth:`Model.collect_regularization_term`."""
        raise RuntimeError

    def score_hrt(self, hrt_batch: torch.LongTensor):  # noqa:D102
        """Non-parametric models do not implement :meth:`Model.score_hrt`."""
        raise RuntimeError

    def score_r(self, ht_batch: torch.LongTensor):  # noqa:D102
        """Non-parametric models do not implement :meth:`Model.score_r`."""
        raise RuntimeError


class MarginalDistributionBaseline(EvaluationOnlyModel):
    r"""
    Score based on marginal distributions.

    To predict scores for the tails, we make the following simplification of $P(t | h, r)$:

    .. math ::
        P(t | h, r) \sim P(t | h) * P(t | r)

    Depending on the settings, we either set $P(t | *) = \frac{1}{n}$ where $n$ is the number of entities,
    or estimate them by counting occurrences in the training triples.

    .. note ::
        This model cannot make use of GPU acceleration, since internally it uses scipy's sparse matrices.

    ---
    name: Marginal Distribution Baseline
    citation:
        author: Berrendorf
        year: 2021
        link: https://github.com/pykeen/pykeen/pull/514
        github: pykeen/pykeen
    """

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        entity_margin: bool = True,
        relation_margin: bool = True,
    ):
        """
        Initialize the model.

        :param triples_factory:
            The triples factory containing the training triples.
        :param entity_margin:
            ...
        :param relation_margin:
            ...

        If you set ``entity_margin=False`` and ``relation_margin=False``, it will
        lead to a uniform distribution, i.e. equal scores for all entities.
        """
        super().__init__(triples_factory=triples_factory)
        h, r, t = numpy.asarray(triples_factory.mapped_triples).T
        if relation_margin:
            self.head_per_relation, self.tail_per_relation = [
                get_csr_matrix(
                    row_indices=r,
                    col_indices=col_indices,
                    shape=(triples_factory.num_relations, triples_factory.num_entities),
                )
                for col_indices in (h, t)
            ]
        else:
            self.head_per_relation = self.tail_per_relation = None
        if entity_margin:
            self.head_per_tail, self.tail_per_head = [
                get_csr_matrix(
                    row_indices=row_indices,
                    col_indices=col_indices,
                    shape=(triples_factory.num_entities, triples_factory.num_entities),
                )
                for row_indices, col_indices in ((t, h), (h, t))
            ]
        else:
            self.head_per_tail = self.tail_per_head = None

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        return marginal_score(
            entity_relation_batch=hr_batch,
            per_entity=self.tail_per_head,
            per_relation=self.tail_per_relation,
            num_entities=self.num_entities,
        )

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        return marginal_score(
            entity_relation_batch=rt_batch.flip(1),
            per_entity=self.head_per_tail,
            per_relation=self.head_per_relation,
            num_entities=self.num_entities,
        )
