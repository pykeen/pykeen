"""Filtered models."""

from typing import Collection, List, Optional, Union

import numpy
import scipy.sparse
import torch
from class_resolver import HintOrType
from docdata import parse_docdata

from pykeen.evaluation.evaluator import prepare_filter_triples

from ..base import Model
from ..baseline.utils import get_csr_matrix
from ...triples.triples_factory import CoreTriplesFactory
from ...typing import InductiveMode, MappedTriples

__all__ = [
    "CooccurrenceFilteredModel",
]


@parse_docdata
class CooccurrenceFilteredModel(Model):
    """A model which filters predictions by co-occurence.

    ---
    citation:
        author: Berrendorf
        year: 2022
        link: https://github.com/pykeen/pykeen/pull/943
        github: pykeen/pykeen
    """

    head_per_relation: scipy.sparse.csr_matrix
    tail_per_relation: scipy.sparse.csr_matrix
    relation_per_head: scipy.sparse.csr_matrix
    relation_per_tail: scipy.sparse.csr_matrix

    TRAINING_FILL_VALUE: float = -1.0e03
    INFERENCE_FILL_VALUE: float = float("-inf")

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        additional_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
        apply_in_training: bool = False,
        base: HintOrType[Model] = "rotate",
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the (training) triples factory; used for creating the co-occurrence counts *and* for instantiating the
            base model.
        :param additional_filter_triples:
            additional triples to use for creating the co-occurrence statistics
        :param in_training:
            whether to apply the masking also during training
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
        h, r, t = (
            prepare_filter_triples(
                mapped_triples=triples_factory.mapped_triples, additional_filter_triples=additional_triples or []
            )
            .numpy()
            .T
        )
        self.head_per_relation, self.tail_per_relation = [
            get_csr_matrix(
                row_indices=r,
                col_indices=col_indices,
                shape=(triples_factory.num_relations, triples_factory.num_entities),
            ).astype(bool)
            for col_indices in (h, t)
        ]
        self.relation_per_head, self.relation_per_tail = [
            m.transpose().tocsr() for m in (self.head_per_relation, self.tail_per_relation)
        ]

        self.apply_in_training = apply_in_training

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
        if self.apply_in_training:
            raise NotImplementedError
        return self.base.score_hrt(hrt_batch=hrt_batch, mode=mode)

    # docstr-coverage: inherited
    def score_h(
        self, rt_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:  # noqa: D102
        return self._mask(
            scores=self.base.score_h(rt_batch=rt_batch, slice_size=slice_size, mode=mode),
            batch_indices=rt_batch[:, 0],
            index=self.head_per_relation,
            in_training=True,
        )

    # docstr-coverage: inherited
    def score_r(
        self, ht_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:  # noqa: D102
        return self._mask(
            self._mask(
                scores=self.base.score_r(ht_batch=ht_batch, slice_size=slice_size, mode=mode),
                batch_indices=ht_batch[:, 0],
                index=self.relation_per_head,
                in_training=True,
            ),
            batch_indices=ht_batch[:, 1],
            index=self.relation_per_tail,
            in_training=True,
        )

    # docstr-coverage: inherited
    def score_t(
        self, hr_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:  # noqa: D102
        return self._mask(
            scores=self.base.score_t(hr_batch=hr_batch, slice_size=slice_size, mode=mode),
            batch_indices=hr_batch[:, 1],
            index=self.head_per_relation,
            in_training=True,
        )

    def _mask(
        self,
        scores: torch.FloatTensor,
        batch_indices: torch.LongTensor,
        index: scipy.sparse.csr_matrix,
        in_training: bool,
    ) -> torch.Tensor:
        if in_training and not self.apply_in_training:
            return scores
        fill_value = self.TRAINING_FILL_VALUE if in_training else self.INFERENCE_FILL_VALUE
        # get batch indices as numpy array
        i = batch_indices.cpu().numpy()
        # get mask, shape: (batch_size, num_entities/num_relations)
        mask = index[i]
        # get non-zero entries
        rows, cols = [torch.as_tensor(ind, device=scores.device, dtype=torch.long) for ind in mask.nonzero()]
        # set scores for -inf for every non-occuring entry
        new_scores = scores.new_full(size=scores.shape, fill_value=fill_value)
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
            in_training=False,
        )

    # docstr-coverage: inherited
    def predict_t(
        self, hr_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:  # noqa: D102
        return self._mask(
            scores=super().predict_t(hr_batch, slice_size=slice_size, mode=mode),
            batch_indices=hr_batch[:, 1],
            index=self.tail_per_relation,
            in_training=False,
        )

    # docstr-coverage: inherited
    def predict_r(
        self, ht_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:  # noqa: D102
        return self._mask(
            scores=self._mask(
                scores=super().predict_r(ht_batch, slice_size=slice_size, mode=mode),
                batch_indices=ht_batch[:, 0],
                index=self.relation_per_head,
                in_training=False,
            ),
            batch_indices=ht_batch[:, 1],
            index=self.relation_per_tail,
            in_training=False,
        )
