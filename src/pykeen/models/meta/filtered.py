"""Filtered models."""

from typing import Optional

import docdata
import numpy
import scipy.sparse
import torch
from class_resolver import HintOrType

from ..base import Model
from ..baseline.utils import get_csr_matrix
from ...triples.triples_factory import CoreTriplesFactory
from ...typing import InductiveMode
from docdata import parse_docdata
__all__ = [
    "PseudoTypeFilteredModel",
]

@parse_docdata
class PseudoTypeFilteredModel(Model):
    """A model which filters predictions by pseudo-types.

    ---
    citation:
        author: Berrendorf
        year: 2022
        link: https://github.com/pykeen/pykeen/pull/943
        github: pykeen/pykeen
    """

    def __init__(self, *, triples_factory: CoreTriplesFactory, base: HintOrType[Model] = "rotate", **kwargs) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the (training) triples factory; used for creating the co-occurrence counts *and* for instantiating the
            base model.
        :param base:
            the base model, or a hint thereof.
        """
        # avoid cyclic imports
        from .. import model_resolver

        # create base model
        base = model_resolver.make(base, triples_factory=triples_factory, pos_kwargs=kwargs)

        super().__init__(
            triples_factory=triples_factory,
            loss=base.loss,
            predict_with_sigmoid=base.predict_with_sigmoid,
            random_seed=base._random_seed,
        )
        # assign *after* nn.Module.__init__
        self.base = base

        # index
        h, r, t = numpy.asarray(triples_factory.mapped_triples).T
        # TODO: binary only?
        self.head_per_relation, self.tail_per_relation = [
            get_csr_matrix(
                row_indices=r,
                col_indices=col_indices,
                shape=(triples_factory.num_relations, triples_factory.num_entities),
            )
            for col_indices in (h, t)
        ]

    # docstr-coverage: inherited
    def _get_entity_len(self, *, mode: Optional[InductiveMode]) -> Optional[int]:
        return self.base._get_entity_len(mode=mode)

    # docstr-coverage: inherited
    def _reset_parameters_(self):
        return self.base._reset_parameters_()

    # docstr-coverage: inherited
    def collect_regularization_term(self) -> torch.FloatTensor:  # noqa: D102
        return self.base.collect_regularization_term()

    # docstr-coverage: inherited
    def score_hrt(
        self, hrt_batch: torch.LongTensor, *, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:  # noqa: D102
        return self.base.score_hrt(hrt_batch=hrt_batch, mode=mode)

    # docstr-coverage: inherited
    def score_h(
        self, rt_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:  # noqa: D102
        return self.base.score_h(rt_batch=rt_batch, slice_size=slice_size, mode=mode)

    # docstr-coverage: inherited
    def score_r(
        self, ht_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:  # noqa: D102
        return self.base.score_r(ht_batch=ht_batch, slice_size=slice_size, mode=mode)

    # docstr-coverage: inherited
    def score_t(
        self, hr_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:  # noqa: D102
        return self.base.score_t(hr_batch=hr_batch, slice_size=slice_size, mode=mode)

    @staticmethod
    def _mask(
        scores: torch.FloatTensor, batch_indices: torch.LongTensor, index: scipy.sparse.csr_matrix
    ) -> torch.FloatTensor:
        # get batch indices as numpy array
        i = batch_indices.cpu().numpy()
        # get mask, shape: (batch_size, num_entities/num_relations)
        mask = index[i]
        # get non-zero entries
        rows, cols = [torch.as_tensor(ind, device=scores.device, dtype=torch.long) for ind in mask.nonzero()]
        # set scores for -inf for every non-occuring entry
        new_scores = scores.new_full(size=scores.shape, fill_value=float("-inf"))
        new_scores[rows, cols] = scores[rows, cols]
        return new_scores

    # docstr-coverage: inherited
    def predict_h(
        self, rt_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:  # noqa: D102
        return self._mask(
            scores=super().predict_h(rt_batch, slice_size=slice_size, mode=mode),
            batch_indices=rt_batch[:, 0],
            index=self.head_per_relation,
        )

    # docstr-coverage: inherited
    def predict_t(
        self, hr_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:  # noqa: D102
        return self._mask(
            scores=super().predict_t(hr_batch, slice_size=slice_size, mode=mode),
            batch_indices=hr_batch[:, 1],
            index=self.tail_per_relation,
        )

    # docstr-coverage: inherited
    def predict_r(
        self, ht_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:  # noqa: D102
        raise NotImplementedError
