# -*- coding: utf-8 -*-

"""A wrapper which combines an interaction function with NodePiece entity representations."""

import logging
from typing import Any, Callable, ClassVar, Mapping, Optional

import torch
from class_resolver import Hint, HintOrType, OptionalKwargs

from .base import InductiveERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import (
    DistMultInteraction,
    Interaction,
    NodePieceRepresentation,
    SubsetRepresentation,
    representation_resolver,
)
from ...nn.node_piece import RelationTokenizer
from ...triples.triples_factory import CoreTriplesFactory

__all__ = [
    "InductiveNodePiece",
]

logger = logging.getLogger(__name__)


class InductiveNodePiece(InductiveERModel):
    """A wrapper which combines an interaction function with NodePiece entity representations from [galkin2021]_.

    This model uses the :class:`pykeen.nn.NodePieceRepresentation` instead of a typical
    :class:`pykeen.nn.Embedding` to more efficiently store representations.
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
        validation_factory: Optional[CoreTriplesFactory] = None,
        test_factory: Optional[CoreTriplesFactory] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the triples factory of training triples. Must have create_inverse_triples set to True.
        :param inference_factory:
            the triples factory of inference triples. Must have create_inverse_triples set to True.
        :param validation_factory:
            the triples factory of validation triples. Must have create_inverse_triples set to True.
        :param test_factory:
            the triples factory of testing triples. Must have create_inverse_triples set to True.
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

        # always create representations for normal and inverse relations and padding
        relation_representations = representation_resolver.make(
            query=None,
            pos_kwargs=relation_representations_kwargs,
            max_id=2 * triples_factory.real_num_relations + 1,
            shape=embedding_dim,
        )
        if validation_factory is None:
            validation_factory = inference_factory

        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations=NodePieceRepresentation,
            entity_representations_kwargs=dict(
                triples_factory=triples_factory,
                tokenizers=RelationTokenizer,
                token_representations=relation_representations,
                aggregation=aggregation,
                num_tokens=num_tokens,
            ),
            relation_representations=SubsetRepresentation(  # hide padding relation
                max_id=triples_factory.num_relations,
                base=relation_representations,
            ),
            validation_factory=validation_factory,
            testing_factory=test_factory,
            **kwargs,
        )
        # note: we need to share the aggregation across representations, since the aggregation may have
        #   trainable parameters
        np: NodePieceRepresentation = self.entity_representations[0]
        for representations in self._mode_to_representations.values():
            assert len(representations) == 1
            np2 = representations[0]
            assert isinstance(np2, NodePieceRepresentation)
            np2.combination = np.combination
