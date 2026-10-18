"""Filtered models."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import scipy.sparse
import torch
from class_resolver import HintOrType

from ..base import Model
from ..baseline.utils import get_csr_matrix
from ...constants import TARGET_TO_INDEX
from ...triples.triples_factory import CoreTriplesFactory
from ...typing import (
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    TRAINING,
    VALIDATION,
    FloatTensor,
    InductiveMode,
    LongTensor,
    MappedTriples,
    Target,
)
from ...utils import prepare_filter_triples

__all__ = [
    "CooccurrenceFilteredModel",
]

#: the indexed triples of a graph, i.e., sparse co-occurrence masks for each pair of (column, row) targets
Indexes = Mapping[Target, Mapping[Target, scipy.sparse.csr_matrix]]


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
    hpo_default: ClassVar[Mapping[str, Any]] = {
        "base": {"type": "categorical", "choices": ["distmult", "mure", "rescal", "rotate", "transe"]},
        "conjunctive": {"type": bool},
    }

    #: the indexed filter triples, i.e., sparse masks, using the "real" relation IDs (for prediction)
    indexes: Indexes

    #: the indexed filter triples using the model's internal relation IDs (for scoring), i.e., including the
    #: inverse triples if the model uses inverse triples; otherwise, identical to :attr:`indexes`
    training_indexes: Indexes

    #: the pair of (:attr:`indexes`, :attr:`training_indexes`) analogues for the inference graph, which is used in the
    #: inductive modes "validation" and "testing"; None in the transductive setting
    inference_indexes: tuple[Indexes, Indexes] | None

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        additional_triples: MappedTriples | list[MappedTriples] | None = None,
        apply_in_training: bool = False,
        base: HintOrType[Model] = "rotate",
        training_fill_value: float = -1.0e03,
        inference_fill_value: float = float("-inf"),
        conjunctive: bool = False,
        inference_factory: CoreTriplesFactory | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the (training) triples factory; used for creating the co-occurrence counts *and* for instantiating the
            base model.
        :param additional_triples:
            additional triples to use for creating the co-occurrence statistics of the training graph
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
        :param inference_factory:
            the inference graph's triples factory for inductive base models; used for creating the co-occurrence
            masks for the modes "validation" and "testing", whose triples are both queries against this graph. If the
            base model is instantiated here, it is also passed on to it.
        :param kwargs:
            additional keyword-based parameters passed to the base model upon instantiation
        """
        # avoid cyclic imports
        from .. import model_resolver

        # create base model; an already instantiated base model has to be given the inference factory itself
        if inference_factory is not None and not isinstance(base, Model):
            kwargs["inference_factory"] = inference_factory
        base = model_resolver.make(base, triples_factory=triples_factory, pos_kwargs=kwargs)

        super().__init__(
            triples_factory=triples_factory,
            loss=base.loss,
            predict_with_sigmoid=base.predict_with_sigmoid,
            random_seed=base._random_seed,
            use_inverse_triples=base.use_inverse_triples,
        )
        # assign *after* nn.Module.__init__
        self.base = base

        # save constants
        self.conjunctive = conjunctive
        self.apply_in_training = apply_in_training
        self.training_fill_value = training_fill_value
        self.inference_fill_value = inference_fill_value

        # index triples of the training graph
        mapped_triples = prepare_filter_triples(
            mapped_triples=triples_factory.mapped_triples, additional_filter_triples=additional_triples, warn=False
        )
        self.indexes, self.training_indexes = self._build_graph_indexes(
            mapped_triples=mapped_triples, num_entities=self.num_entities
        )

        # index triples of the inference graph (inductive setting)
        self.inference_indexes = (
            None
            if inference_factory is None
            # validation and testing triples are both queries against the inference graph
            else self._build_graph_indexes(
                mapped_triples=inference_factory.mapped_triples, num_entities=self._get_entity_len(mode=VALIDATION)
            )
        )

        # initialize base model's parameters
        self.reset_parameters_()

    def _build_graph_indexes(self, mapped_triples: MappedTriples, num_entities: int) -> tuple[Indexes, Indexes]:
        """Index the triples of a graph, using the "real" and the model's internal relation IDs, respectively."""
        # the prediction methods use the "real" relation IDs
        indexes = self._build_indexes(
            mapped_triples=mapped_triples.numpy(), nums=[num_entities, self.num_real_relations, num_entities]
        )
        if not self.use_inverse_triples:
            # without inverse relations, the real and the internal relation IDs coincide
            return indexes, indexes
        # the scoring methods (used during training) operate on the model's internal relation IDs, which comprise
        # the inverse relations, whereas the prediction methods use the real relation IDs. Hence, the masks differ
        relations = self.relation_inverter.to_internal(mapped_triples[:, 1])
        internal_triples = torch.cat(
            [
                torch.stack([mapped_triples[:, 0], relations, mapped_triples[:, 2]], dim=-1),
                torch.stack(
                    [mapped_triples[:, 2], self.relation_inverter.get_inverse_id(relations), mapped_triples[:, 0]],
                    dim=-1,
                ),
            ]
        )
        training_indexes = self._build_indexes(
            mapped_triples=internal_triples.numpy(), nums=[num_entities, self.num_relations, num_entities]
        )
        return indexes, training_indexes

    @staticmethod
    def _build_indexes(mapped_triples: np.ndarray, nums: list[int]) -> Indexes:
        return {
            col_label: {
                row_label: get_csr_matrix(
                    row_indices=mapped_triples[:, row_index],
                    col_indices=mapped_triples[:, col_index],
                    shape=(num_rows, num_cols),
                    # TODO how to type this properly? np.bool and np.bool_ don't work either
                    dtype=bool,  # type: ignore[arg-type]
                    norm=None,
                )
                for num_rows, (row_label, row_index) in zip(nums, TARGET_TO_INDEX.items(), strict=False)
                if row_label != col_label
            }
            for num_cols, (col_label, col_index) in zip(nums, TARGET_TO_INDEX.items(), strict=False)
        }

    def _mask(
        self,
        scores: FloatTensor,
        batch: LongTensor,
        target: Target,
        in_training: bool,
        mode: InductiveMode | None,
        ids: LongTensor | None = None,
    ) -> FloatTensor:
        if in_training and not self.apply_in_training:
            return scores
        fill_value = self.training_fill_value if in_training else self.inference_fill_value
        indexes = self._get_indexes(mode=mode, in_training=in_training)
        # get masks, shape: (batch_size, num_entities/num_relations)
        first_mask, second_mask = (
            index[batch_indices.cpu().numpy()]
            for batch_indices, (_target, index) in zip(
                batch.t(), sorted(indexes[target].items(), key=lambda kv: TARGET_TO_INDEX[kv[0]]), strict=False
            )
        )
        # combine masks
        # note: for sparse matrices, * is a matrix product; .multiply is an elementwise and, and + an elementwise or
        mask = first_mask.multiply(second_mask) if self.conjunctive else (first_mask + second_mask)
        dense_mask = torch.as_tensor(mask.toarray(), device=scores.device, dtype=torch.bool)
        # restrict to the scored ids, shape: (num_ids,) or (batch_size, num_ids)
        if ids is not None:
            ids = ids.to(device=scores.device)
            dense_mask = dense_mask[:, ids] if ids.ndim == 1 else dense_mask.gather(dim=1, index=ids)
        # set scores for fill value for every non-occuring entry
        return scores.masked_fill(~dense_mask, fill_value)

    def _get_entity_len(self, *, mode: InductiveMode | None) -> int:
        return self.base._get_entity_len(mode=mode)

    def _reset_parameters_(self):
        return self.base._reset_parameters_()

    def collect_regularization_term(self) -> FloatTensor:  # noqa: D102
        return self.base.collect_regularization_term()

    def _get_indexes(self, mode: InductiveMode | None, in_training: bool) -> Indexes:
        """Select the indexes of the graph corresponding to the mode, using real or internal relation IDs."""
        if mode is None or mode == TRAINING:
            indexes, training_indexes = self.indexes, self.training_indexes
        elif self.inference_indexes is not None:
            # the validation and testing triples are queries against the inference graph
            indexes, training_indexes = self.inference_indexes
        else:
            raise ValueError(
                f"{self.__class__.__name__} has no inference graph for mode={mode}; pass `inference_factory`."
            )
        return training_indexes if in_training else indexes

    def score_hrt(self, hrt_batch: LongTensor, **kwargs) -> FloatTensor:  # noqa: D102
        if self.apply_in_training:
            raise NotImplementedError
        return self.base.score_hrt(hrt_batch=hrt_batch, **kwargs)

    def score_h(  # noqa: D102
        self, rt_batch: LongTensor, *, heads: LongTensor | None = None, mode: InductiveMode | None = None, **kwargs
    ) -> FloatTensor:
        return self._mask(
            scores=self.base.score_h(rt_batch=rt_batch, heads=heads, mode=mode, **kwargs),
            batch=rt_batch,
            target=LABEL_HEAD,
            in_training=True,
            mode=mode,
            ids=heads,
        )

    def score_r(  # noqa: D102
        self, ht_batch: LongTensor, *, relations: LongTensor | None = None, mode: InductiveMode | None = None, **kwargs
    ) -> FloatTensor:
        return self._mask(
            scores=self.base.score_r(ht_batch=ht_batch, relations=relations, mode=mode, **kwargs),
            batch=ht_batch,
            target=LABEL_RELATION,
            in_training=True,
            mode=mode,
            ids=relations,
        )

    def score_t(  # noqa: D102
        self, hr_batch: LongTensor, *, tails: LongTensor | None = None, mode: InductiveMode | None = None, **kwargs
    ) -> FloatTensor:
        return self._mask(
            scores=self.base.score_t(hr_batch=hr_batch, tails=tails, mode=mode, **kwargs),
            batch=hr_batch,
            target=LABEL_TAIL,
            in_training=True,
            mode=mode,
            ids=tails,
        )

    def predict_h(  # noqa: D102
        self, rt_batch: LongTensor, *, heads: LongTensor | None = None, mode: InductiveMode | None = None, **kwargs
    ) -> FloatTensor:
        return self._mask(
            scores=super().predict_h(rt_batch, heads=heads, mode=mode, **kwargs),
            batch=rt_batch,
            target=LABEL_HEAD,
            in_training=False,
            mode=mode,
            ids=heads,
        )

    def predict_t(  # noqa: D102
        self, hr_batch: LongTensor, *, tails: LongTensor | None = None, mode: InductiveMode | None = None, **kwargs
    ) -> FloatTensor:
        return self._mask(
            scores=super().predict_t(hr_batch, tails=tails, mode=mode, **kwargs),
            batch=hr_batch,
            target=LABEL_TAIL,
            in_training=False,
            mode=mode,
            ids=tails,
        )

    def predict_r(  # noqa: D102
        self, ht_batch: LongTensor, *, relations: LongTensor | None = None, mode: InductiveMode | None = None, **kwargs
    ) -> FloatTensor:
        return self._mask(
            scores=super().predict_r(ht_batch, relations=relations, mode=mode, **kwargs),
            batch=ht_batch,
            target=LABEL_RELATION,
            in_training=False,
            mode=mode,
            ids=relations,
        )
