# -*- coding: utf-8 -*-

"""A wrapper which combines an interaction function with NodePiece entity representations."""

import logging
from typing import Any, Callable, ClassVar, Mapping, Optional, Sequence, Tuple, Union

import torch
from class_resolver import Hint, HintOrType
from torch import nn

from .node_piece import _ConcatMLP
from ..nbase import ERModel, _prepare_representation_module_list
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.emb import EmbeddingSpecification, NodePieceRepresentation, SubsetRepresentationModule
from ...nn.modules import DistMultInteraction, Interaction
from ...triples.triples_factory import CoreTriplesFactory
from ...typing import HeadRepresentation, Mode, RelationRepresentation, TailRepresentation, cast

__all__ = [
    "InductiveNodePiece",
]

logger = logging.getLogger(__name__)


class InductiveNodePiece(ERModel):
    """A wrapper which combines an interaction function with NodePiece entity representations from [galkin2021]_.

    This model uses the :class:`pykeen.nn.emb.NodePieceRepresentation` instead of a typical
    :class:`pykeen.nn.emb.Embedding` to more efficiently store representations.

    INDUCTIVE VERSION
    ---
    citation:
        author: Galkin
        year: 2021
        link: https://arxiv.org/abs/2106.12144
        github: https://github.com/migalkin/NodePiece
    """

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        inference_factory: CoreTriplesFactory,
        num_tokens: int = 2,
        embedding_dim: int = 64,
        embedding_specification: Optional[EmbeddingSpecification] = None,
        interaction: HintOrType[Interaction] = DistMultInteraction,
        aggregation: Hint[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        shape: Optional[Sequence[int]] = None,
        validation_factory: Optional[CoreTriplesFactory] = None,
        test_factory: Optional[CoreTriplesFactory] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the triples factory. Must have create_inverse_triples set to True.
        :param num_tokens:
            the number of relations to use to represent each entity, cf.
            :class:`pykeen.nn.emb.NodePieceRepresentation`.
        :param embedding_dim:
            the embedding dimension. Only used if embedding_specification is not given.
        :param embedding_specification:
            the embedding specification.
        :param interaction:
            the interaction module, or a hint for it.
        :param aggregation:
            aggregation of multiple token representations to a single entity representation. By default,
            this uses :func:`torch.mean`. If a string is provided, the module assumes that this refers to a top-level
            torch function, e.g. "mean" for :func:`torch.mean`, or "sum" for func:`torch.sum`. An aggregation can
            also have trainable parameters, .e.g., ``MLP(mean(MLP(tokens)))`` (cf. DeepSets from [zaheer2017]_). In
            this case, the module has to be created outside of this component.

            Moreover, we support providing "mlp" as a shortcut to use the MLP aggregation version from [galkin2021]_.

            We could also have aggregations which result in differently shapes output, e.g. a concatenation of all
            token embeddings resulting in shape ``(num_tokens * d,)``. In this case, `shape` must be provided.

            The aggregation takes two arguments: the (batched) tensor of token representations, in shape
            ``(*, num_tokens, *dt)``, and the index along which to aggregate.
        :param shape:
            the shape of an individual representation. Only necessary, if aggregation results in a change of dimensions.
            this will only be necessary if the aggregation is an *ad hoc* function.
        :param kwargs:
            additional keyword-based arguments passed to :meth:`ERModel.__init__`

        :raises ValueError:
            if the triples factory does not create inverse triples
        """
        if not triples_factory.create_inverse_triples:
            raise ValueError(
                "The provided triples factory does not create inverse triples. However, for the node piece "
                "representations inverse relation representations are required.",
            )
        embedding_specification = embedding_specification or EmbeddingSpecification(
            shape=(embedding_dim,),
        )

        # Create an MLP for string aggregation
        if aggregation == "mlp":
            aggregation = _ConcatMLP(
                num_tokens=num_tokens,
                embedding_dim=embedding_dim,
            )

        # always create representations for normal and inverse relations and padding
        relation_representations = embedding_specification.make(
            num_embeddings=2 * triples_factory.real_num_relations + 1,
        )
        entity_representations = NodePieceRepresentation(
            triples_factory=triples_factory,
            token_representation=relation_representations,
            aggregation=aggregation,
            shape=shape,
            num_tokens=num_tokens,
        )

        inference_representation = NodePieceRepresentation(
            triples_factory=inference_factory,
            token_representation=relation_representations,
            aggregation=aggregation,
            shape=shape,
            num_tokens=num_tokens,
        )

        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations=entity_representations,
            relation_representations=SubsetRepresentationModule(  # hide padding relation
                relation_representations,
                max_id=triples_factory.num_relations,
            ),
            **kwargs,
        )

        self.inference_representation = _prepare_representation_module_list(
            representations=inference_representation,
            num_embeddings=inference_factory.num_entities,
            shapes=self.interaction.entity_shape,
            label="entity",
            skip_checks=self.interaction.tail_entity_shape is not None or kwargs["skip_checks"]
            if "skip_checks" in kwargs
            else False,
        )

        self.num_train_entities = triples_factory.num_entities
        self.num_inference_entities, self.num_valid_entities, self.num_test_entities = None, None, None
        if inference_factory is not None:
            self.num_inference_entities = inference_factory.num_entities
            self.num_valid_entities = self.num_test_entities = self.num_inference_entities
        else:
            self.num_valid_entities = validation_factory.num_entities
            self.num_test_entities = test_factory.num_entities

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
        mode: Mode,
        slice_size: Optional[int] = None,
        slice_dim: Optional[str] = None,
    ) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail indices and calculates the corresponding score.

        All indices which are not None, have to be either 1-element or have the same shape, which is the batch size.

        :param h_indices:
            The head indices. None indicates to use all.
        :param r_indices:
            The relation indices. None indicates to use all.
        :param t_indices:
            The tail indices. None indicates to use all.
        :param slice_size:
            The slice size.
        :param slice_dim:
            The dimension along which to slice. From {"h", "r", "t"}

        :return: shape: (batch_size, num_heads, num_relations, num_tails)
            The score for each triple.
        """
        h, r, t = self._get_representations(h_indices=h_indices, r_indices=r_indices, t_indices=t_indices, mode=mode)
        scores = self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=slice_dim)
        return self._repeat_scores_if_necessary(
            scores=scores,
            h_indices=h_indices,
            r_indices=r_indices,
            t_indices=t_indices,
        )

    def _get_representations(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
        mode: Mode,
    ) -> Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation]:
        """Get representations for head, relation and tails, in canonical shape."""
        entity_representations = self.entity_representations if mode == "train" else self.inference_representation

        h, r, t = [
            [representation.get_in_more_canonical_shape(dim=dim, indices=indices) for representation in representations]
            for dim, indices, representations in (
                ("h", h_indices, entity_representations),
                ("r", r_indices, self.relation_representations),
                ("t", t_indices, entity_representations),
            )
        ]
        # normalization
        return cast(
            Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation],
            tuple(x[0] if len(x) == 1 else x for x in (h, r, t)),
        )

    def score_hrt(self, hrt_batch: torch.LongTensor, mode: Mode = "train") -> torch.FloatTensor:
        return self(
            h_indices=hrt_batch[:, 0],
            r_indices=hrt_batch[:, 1],
            t_indices=hrt_batch[:, 2],
            mode=mode,
        ).view(hrt_batch.shape[0], 1)

    def score_t(
        self, hr_batch: torch.LongTensor, slice_size: Optional[int] = None, mode: Mod = "train"
    ) -> torch.FloatTensor:
        return self(
            h_indices=hr_batch[:, 0],
            r_indices=hr_batch[:, 1],
            t_indices=None,
            slice_size=slice_size,
            slice_dim="h",
            mode=mode,
        ).view(hr_batch.shape[0], getattr(self, f"num_{mode}_entities"))

    def score_h_inverse(self, rt_batch: torch.LongTensor, mode: Mode, slice_size: Optional[int] = None):
        """Score all heads for a batch of (r,t)-pairs using the tail predictions for the inverses $(t,r_{inv},*)$."""
        t_r_inv = self._prepare_inverse_batch(batch=rt_batch, index_relation=0)

        if slice_size is None:
            return self.score_t(hr_batch=t_r_inv, mode=mode)
        else:
            return self.score_t(hr_batch=t_r_inv, mode=mode, slice_size=slice_size)  # type: ignore

    # for evaluation
    def predict_h(
        self,
        rt_batch: torch.LongTensor,
        mode: Mode,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        self.eval()  # Enforce evaluation mode
        rt_batch = self._prepare_batch(batch=rt_batch, index_relation=0)
        if self.use_inverse_triples:
            scores = self.score_h_inverse(rt_batch=rt_batch, mode=mode, slice_size=slice_size)
        elif slice_size is None:
            scores = self.score_h(rt_batch, mode=mode)
        else:
            scores = self.score_h(rt_batch, mode=mode, slice_size=slice_size)  # type: ignore
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores

    def predict_t(
        self,
        hr_batch: torch.LongTensor,
        mode: Mode,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        self.eval()  # Enforce evaluation mode
        hr_batch = self._prepare_batch(batch=hr_batch, index_relation=1)
        if slice_size is None:
            scores = self.score_t(hr_batch, mode=mode)
        else:
            scores = self.score_t(hr_batch, mode=mode, slice_size=slice_size)  # type: ignore
        if self.predict_with_sigmoid:
            scores = torch.sigmoid(scores)
        return scores
