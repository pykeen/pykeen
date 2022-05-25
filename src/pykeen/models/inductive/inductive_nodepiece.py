# -*- coding: utf-8 -*-

"""A wrapper which combines an interaction function with NodePiece entity representations."""

import logging
from typing import Any, Callable, ClassVar, Mapping, Optional, Sequence

import torch
from class_resolver import Hint, HintOrType, OptionalKwargs

from ..nbase import ERModel, _prepare_representation_module_list
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import (
    DistMultInteraction,
    Interaction,
    NodePieceRepresentation,
    Representation,
    SubsetRepresentation,
    representation_resolver,
)
from ...nn.node_piece import RelationTokenizer
from ...nn.perceptron import ConcatMLP
from ...triples.triples_factory import CoreTriplesFactory
from ...typing import TESTING, TRAINING, VALIDATION, InductiveMode, OneOrSequence

__all__ = [
    "InductiveNodePiece",
]

logger = logging.getLogger(__name__)


class InductiveNodePiece(ERModel):
    """A wrapper which combines an interaction function with NodePiece entity representations from [galkin2021]_.

    This model uses the :class:`pykeen.nn.NodePieceRepresentation` instead of a typical
    :class:`pykeen.nn.Embedding` to more efficiently store representations.

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
        relation_representations_kwargs: OptionalKwargs = None,
        interaction: HintOrType[Interaction] = DistMultInteraction,
        aggregation: Hint[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        shape: Optional[OneOrSequence[int]] = None,
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
            :class:`pykeen.nn.NodePieceRepresentation`.
        :param embedding_dim:
            the embedding dimension. Only used if embedding_specification is not given.
        :param relation_representations_kwargs:
            the relation representation parameters
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

        # Create an MLP for string aggregation
        if aggregation == "mlp":
            aggregation = ConcatMLP(
                num_tokens=num_tokens,
                embedding_dim=embedding_dim,
            )

        # always create representations for normal and inverse relations and padding
        relation_representations = representation_resolver.make(
            query=None,
            pos_kwargs=relation_representations_kwargs,
            max_id=2 * triples_factory.real_num_relations + 1,
            shape=embedding_dim,
        )

        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations=NodePieceRepresentation,
            entity_representations_kwargs=dict(
                triples_factory=triples_factory,
                tokenizers=RelationTokenizer,
                token_representations=relation_representations,
                aggregation=aggregation,
                shape=shape,
                num_tokens=num_tokens,
            ),
            relation_representations=SubsetRepresentation(  # hide padding relation
                max_id=triples_factory.num_relations,
                base=relation_representations,
            ),
            **kwargs,
        )
        self.inference_representation = _prepare_representation_module_list(
            representations=NodePieceRepresentation,
            representation_kwargs=dict(
                triples_factory=inference_factory,
                tokenizers=RelationTokenizer,
                token_representations=relation_representations,
                aggregation=aggregation,
                shape=shape,
                num_tokens=num_tokens,
            ),
            max_id=inference_factory.num_entities,
            shapes=self.interaction.full_entity_shapes(),
            label="entity",
        )

        self.num_train_entities = triples_factory.num_entities
        self.num_inference_entities, self.num_valid_entities, self.num_test_entities = None, None, None
        if inference_factory is not None:
            self.num_inference_entities = inference_factory.num_entities
            self.num_valid_entities = self.num_test_entities = self.num_inference_entities
        else:
            self.num_valid_entities = validation_factory.num_entities
            self.num_test_entities = test_factory.num_entities

    # docstr-coverage: inherited
    def _get_entity_representations_from_inductive_mode(
        self, *, mode: Optional[InductiveMode]
    ) -> Sequence[Representation]:  # noqa: D102
        if mode == TRAINING:
            return self.entity_representations
        elif mode == TESTING or mode == VALIDATION:
            return self.inference_representation
        elif mode is None:
            raise ValueError(f"{self.__class__.__name__} does not support inductive mode: {mode}")
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # docstr-coverage: inherited
    def _get_entity_len(self, *, mode: Optional[InductiveMode]) -> Optional[int]:  # noqa: D102
        if mode == TRAINING:
            return self.num_train_entities
        elif mode == TESTING:
            return self.num_test_entities
        elif mode == VALIDATION:
            return self.num_valid_entities
        else:
            raise ValueError
