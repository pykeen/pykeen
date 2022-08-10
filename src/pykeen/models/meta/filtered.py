"""Filtered models."""

from typing import Any, ClassVar, List, Mapping, Optional, Union

import scipy.sparse
import torch
from class_resolver import HintOrType
from docdata import parse_docdata

from ..base import Model
from ..baseline.utils import get_csr_matrix
from ...constants import TARGET_TO_INDEX
from ...triples.triples_factory import CoreTriplesFactory
from ...typing import LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, InductiveMode, MappedTriples, Target
from ...utils import prepare_filter_triples

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

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        base=dict(type="categorical", choices=["distmult", "mure", "rescal", "rotate", "transe"]),
        conjunctive=dict(type=bool),
    )

    #: the indexed filter triples, i.e., sparse masks
    indexes: Mapping[Target, Mapping[Target, scipy.sparse.csr_matrix]]

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        additional_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
        apply_in_training: bool = False,
        base: HintOrType[Model] = "rotate",
        training_fill_value: float = -1.0e03,
        inference_fill_value: float = float("-inf"),
        conjunctive: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the (training) triples factory; used for creating the co-occurrence counts *and* for instantiating the
            base model.
        :param additional_triples:
            additional triples to use for creating the co-occurrence statistics
        :param apply_in_training:
            whether to apply the masking also during training
        :param base:
            the base model, or a hint thereof.
        :param training_fill_value:
            the training fill value; for most loss functions, this has to be a finite value, i.e., not infinity
        :param inference_fill_value:
            the inference fill value
        :param conjunctive:
            whether to use conjuction or disjunction to combine non-filter masks
        :param kwargs:
            additional keyword-based parameters passed to the base model upon instantiation
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

        # save constants
        self.conjunctive = conjunctive
        self.apply_in_training = apply_in_training
        self.training_fill_value = training_fill_value
        self.inference_fill_value = inference_fill_value

        # index triples
        mapped_triples = prepare_filter_triples(
            mapped_triples=triples_factory.mapped_triples, additional_filter_triples=additional_triples, warn=False
        ).numpy()
        nums = [triples_factory.num_entities, triples_factory.num_relations, triples_factory.num_entities]
        self.indexes = {
            col_label: {
                row_label: get_csr_matrix(
                    row_indices=mapped_triples[:, row_index],
                    col_indices=mapped_triples[:, col_index],
                    shape=(num_rows, num_cols),
                    dtype=bool,
                    norm=None,
                )
                for num_rows, (row_label, row_index) in zip(nums, TARGET_TO_INDEX.items())
                if row_label != col_label
            }
            for num_cols, (col_label, col_index) in zip(nums, TARGET_TO_INDEX.items())
        }

        # initialize base model's parameters
        self.reset_parameters_()

    def _mask(
        self,
        scores: torch.FloatTensor,
        batch: torch.LongTensor,
        target: Target,
        in_training: bool,
    ) -> torch.FloatTensor:
        if in_training and not self.apply_in_training:
            return scores
        fill_value = self.training_fill_value if in_training else self.inference_fill_value
        # get masks, shape: (batch_size, num_entities/num_relations)
        first_mask, second_mask = [
            index[batch_indices.cpu().numpy()]
            for batch_indices, (_target, index) in zip(
                batch.t(), sorted(self.indexes[target].items(), key=lambda kv: TARGET_TO_INDEX[kv[0]])
            )
        ]
        # combine masks
        # note: * is an elementwise and, and + and elementwise or
        mask = (first_mask * second_mask) if self.conjunctive else (first_mask + second_mask)
        # get non-zero entries
        rows, cols = [torch.as_tensor(ind, device=scores.device, dtype=torch.long) for ind in mask.nonzero()]
        # set scores for fill value for every non-occuring entry
        new_scores = scores.new_full(size=scores.shape, fill_value=fill_value)
        new_scores[rows, cols] = scores[rows, cols]
        return new_scores

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
    def score_hrt(self, hrt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        if self.apply_in_training:
            raise NotImplementedError
        return self.base.score_hrt(hrt_batch=hrt_batch, **kwargs)

    # docstr-coverage: inherited
    def score_h(self, rt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        return self._mask(
            scores=self.base.score_h(rt_batch=rt_batch, **kwargs),
            batch=rt_batch,
            target=LABEL_HEAD,
            in_training=True,
        )

    # docstr-coverage: inherited
    def score_r(self, ht_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        return self._mask(
            scores=self.base.score_r(ht_batch=ht_batch, **kwargs),
            batch=ht_batch,
            target=LABEL_RELATION,
            in_training=True,
        )

    # docstr-coverage: inherited
    def score_t(self, hr_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        return self._mask(
            scores=self.base.score_t(hr_batch=hr_batch, **kwargs),
            batch=hr_batch,
            target=LABEL_TAIL,
            in_training=True,
        )

    # docstr-coverage: inherited
    def predict_h(self, rt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        return self._mask(
            scores=super().predict_h(rt_batch, **kwargs),
            batch=rt_batch,
            target=LABEL_HEAD,
            in_training=False,
        )

    # docstr-coverage: inherited
    def predict_t(self, hr_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        return self._mask(
            scores=super().predict_t(hr_batch, **kwargs),
            batch=hr_batch,
            target=LABEL_TAIL,
            in_training=False,
        )

    # docstr-coverage: inherited
    def predict_r(self, ht_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        return self._mask(
            scores=super().predict_r(ht_batch, **kwargs),
            batch=ht_batch,
            target=LABEL_RELATION,
            in_training=False,
        )
