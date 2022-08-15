# -*- coding: utf-8 -*-

"""Implementation of TripleRE."""

from typing import Any, ClassVar, Mapping, Optional, Tuple

from class_resolver import HintOrType, OptionalKwargs
from torch.nn.init import uniform_

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import Embedding, SubsetRepresentation, representation_resolver
from ...nn.modules import TripleREInteraction
from ...nn.node_piece import NodePieceRepresentation, PrecomputedPoolTokenizer, RelationTokenizer, Tokenizer
from ...triples import CoreTriplesFactory
from ...typing import Hint, Initializer

__all__ = [
    "TripleRE",
]


# cf. https://github.com/LongYu-360/TripleRE-Add-NodePiece/blob/994216dcb1d718318384368dd0135477f852c6a4/TripleRE%2BNodepiece/ogb_wikikg2/model.py#L196-L204  # noqa: E501


class TripleRE(ERModel):
    r"""An implementation of TripleRE related to [yu2021]_.

    This implementation does not use NodePieceRepresentations so far.

    ---
    citation:
        author: Yu
        year: 2021
        link: https://vixra.org/abs/2112.0095
        github: https://github.com/LongYu-360/TripleRE-Add-NodePiece/
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        p=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        embedding_dim: int = 128,
        p: int = 1,
        power_norm: bool = False,
        initializer: Hint[Initializer] = uniform_,
        initializer_kwargs: Optional[Mapping[str, Any]] = None,
        anchor_tokenizer: HintOrType[Tokenizer] = PrecomputedPoolTokenizer,
        anchor_tokenizer_kwargs: OptionalKwargs = None,
        num_tokens: Tuple[int, int] = (20, 12),
        node_piece_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        r"""Initialize TripleRE via the :class:`pykeen.nn.modules.TripleREInteraction` interaction.

        :param triples_factory:
            the (training) triples factory
        :param embedding_dim:
            the entity embedding dimension $d$
        :param p:
            the $l_p$ norm.
        :param power_norm:
            should the power norm be used?
        :param initializer:
            the initializer used for anchor and relation representations.
        :param initializer_kwargs:
            additional initializer keyword-based parameters
        :param anchor_tokenizer:
            the anchor tokenizer
        :param anchor_tokenizer_kwargs:
            additional keyword-based parameters passed to the anchor tokenizer.
        :param num_tokens:
            the number of tokens to select for (1) the relation tokenizer and (2) the anchor tokenizer
        :param node_piece_kwargs:
            keyword-based parameters for NodePieceRepresentation
        :param kwargs:
            remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        assert triples_factory.create_inverse_triples
        relation_mid_representation = representation_resolver.make(
            Embedding,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            max_id=2 * triples_factory.real_num_relations + 1,  # inverse relations + padding
            shape=embedding_dim,
        )
        super().__init__(
            triples_factory=triples_factory,
            interaction=TripleREInteraction,
            interaction_kwargs=dict(p=p, power_norm=power_norm),
            entity_representations=NodePieceRepresentation,
            entity_representations_kwargs=dict(
                triples_factory=triples_factory,
                token_representations=[
                    relation_mid_representation,  # for relation-tokens
                    Embedding,  # for anchor nodes
                ],
                # tokenizers:
                # https://github.com/LongYu-360/TripleRE-Add-NodePiece/blob/994216dcb1d718318384368dd0135477f852c6a4/TripleRE%2BNodepiece/run_ogb.py#L273-L292
                token_representations_kwargs=[
                    None,
                    dict(
                        shape=embedding_dim,
                        initializer=initializer,
                        initializer_kwargs=initializer_kwargs,
                    ),
                ],
                tokenizers=[
                    RelationTokenizer,
                    anchor_tokenizer,
                ],
                tokenizers_kwargs=[
                    #   https://github.com/LongYu-360/TripleRE-Add-NodePiece/blob/994216dcb1d718318384368dd0135477f852c6a4/TripleRE%2BNodepiece/run_ogb.py#L86-L88
                    None,
                    anchor_tokenizer_kwargs,
                ],
                num_tokens=num_tokens,
                **(node_piece_kwargs or {}),
            ),
            relation_representations=[None, SubsetRepresentation, None],
            relation_representations_kwargs=[
                dict(
                    shape=embedding_dim,
                    initializer=initializer,
                    initializer_kwargs=initializer_kwargs,
                ),
                dict(  # hide padding relation
                    # max_id=triples_factory.num_relations,  # will get added by ERModel
                    base=relation_mid_representation,
                ),
                dict(
                    shape=embedding_dim,
                    initializer=initializer,
                    initializer_kwargs=initializer_kwargs,
                ),
            ],
            **kwargs,
        )
