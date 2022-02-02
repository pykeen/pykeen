# -*- coding: utf-8 -*-

"""A wrapper which combines an interaction function with NodePiece entity representations."""

import logging
from typing import Any, Callable, ClassVar, Mapping, Optional, Sequence

import torch
from class_resolver import Hint, HintOrType

from .node_piece import _ConcatMLP
from ..nbase import ERModel, _prepare_representation_module_list
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import (
    DistMultInteraction,
    EmbeddingSpecification,
    Interaction,
    NodePieceRepresentation,
    SubsetRepresentationModule,
)
from ...triples.triples_factory import CoreTriplesFactory
from ...typing import Mode

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

    def _entity_representation_from_mode(self, mode: Mode = None):
        if mode == "training":
            return self.entity_representations
        else:
            return self.inference_representation

    def _get_entity_len(self, mode: Mode = None) -> int:
        if mode == "training":
            return self.num_train_entities
        elif mode == "testing":
            return self.num_test_entities
        elif mode == "validation":
            return self.num_valid_entities
        else:
            raise ValueError
