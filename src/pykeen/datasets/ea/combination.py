"""Combination strategies for entity alignment datasets."""
import logging
from abc import abstractmethod
from typing import Dict

import pandas
import torch
from class_resolver import ClassResolver

from ...triples import CoreTriplesFactory, TriplesFactory
from ...typing import EA_SIDE_LEFT, EA_SIDE_RIGHT
from ...utils import format_relative_comparison, get_connected_components

logger = logging.getLogger(__name__)


# TODO: support ID-only graphs


class GraphPairCombinator:
    """A base class for combination of a graph pair into a single graph."""

    @abstractmethod
    def __call__(
        self,
        left: TriplesFactory,
        right: TriplesFactory,
        alignment: pandas.DataFrame,
        **kwargs,
    ) -> TriplesFactory:
        """
        Combine two graphs using the alignment information.

        :param left:
            the triples of the left graph
        :param right:
            the triples of the right graph
        :param alignment: columns: LEFT | RIGHT
            the alignment, i.e., pairs of matching entities
        :param kwargs:
            additional keyword-based parameters passed to :meth:`TriplesFactory.__init__`

        :return:
            a single triples factory comprising the joint graph.
        """
        raise NotImplementedError


class CollapseGraphCombinator(GraphPairCombinator):
    """This combinator merges all matching entity pairs into a single ID."""

    def __call__(
        self,
        left: TriplesFactory,
        right: TriplesFactory,
        alignment: pandas.DataFrame,
        **kwargs,
    ) -> TriplesFactory:  # noqa: D102
        # concatenate (shifted) triples
        mapped_triples = torch.cat(
            [
                left.mapped_triples,
                right.mapped_triples
                + torch.as_tensor([left.num_entities, left.num_relations, left.num_entities]).view(1, 3),
            ]
        )
        # determine connected components regarding the same-as relation (i.e., applies transitivity)
        id_mapping = torch.arange(left.num_entities + right.num_entities)
        for cc in get_connected_components(
            pairs=(
                (
                    left.entity_labeling.label_to_id[row[EA_SIDE_LEFT]],
                    right.entity_labeling.label_to_id[row[EA_SIDE_RIGHT]] + left.num_entities,
                )
                for _, row in alignment.iterrows()
            )
        ):
            cc = list(cc)
            id_mapping[cc] = min(cc)
        # apply id mapping
        h, r, t = mapped_triples.t()
        h, t = id_mapping[h], id_mapping[t]
        # ensure consecutive IDs
        unique, inverse = torch.cat([h, t]).unique(return_inverse=True)
        h, t = inverse.split(split_size_or_sections=2)
        mapped_triples = torch.stack([h, r, t], dim=-1)
        # TODO: keep labeling?
        return CoreTriplesFactory(
            mapped_triples=mapped_triples,
            num_entities=len(unique),
            num_relations=left.num_relations + right.num_relations,
            entity_ids=None,
            relation_ids=None,
            **kwargs,
        )


class ExtraRelationGraphCombinator(GraphPairCombinator):
    """This combinator keeps all entities, but introduces a novel alignment relation."""

    def __call__(
        self,
        left: TriplesFactory,
        right: TriplesFactory,
        alignment: pandas.DataFrame,
        **kwargs,
    ) -> TriplesFactory:  # noqa: D102
        mapped_triples = []
        entity_to_id: Dict[str, int] = {}
        relation_to_id: Dict[str, int] = {}
        entity_offset = relation_offset = 0
        entity_offsets = []
        for side, tf in ((EA_SIDE_LEFT, left), (EA_SIDE_RIGHT, right)):
            mapped_triples.append(
                tf.mapped_triples + torch.as_tensor(data=[entity_offset, relation_offset, entity_offset]).view(1, 3)
            )
            entity_to_id.update((f"{side}:{key}", value + entity_offset) for key, value in tf.entity_to_id.items())
            relation_to_id.update(
                (f"{side}:{key}", value + relation_offset) for key, value in tf.relation_to_id.items()
            )
            entity_offsets.append(entity_offset)
            entity_offset += tf.num_entities
            relation_offset += tf.num_relations

        # extra alignment relation
        relation_to_id["same-as"] = relation_offset
        # filter alignment
        mask = ~(alignment[EA_SIDE_LEFT].isin(left.entity_to_id) & alignment[EA_SIDE_RIGHT].isin(right.entity_to_id))
        if mask.any():
            alignment = alignment.loc[~mask]
            logger.warning(
                f"Dropped {format_relative_comparison(part=mask.sum(), total=alignment.shape[0])} "
                f"alignments due to unknown labels.",
            )
        # map alignment to (new) IDs
        left_id = (
            alignment[EA_SIDE_LEFT].apply(left.entity_to_id.__getitem__) + entity_offsets[0]
        )  # offset should be zero
        right_id = alignment[EA_SIDE_RIGHT].apply(right.entity_to_id.__getitem__) + entity_offsets[1]
        # append alignment triples
        mapped_triples.append(
            torch.stack(
                [
                    torch.as_tensor(data=left_id.values, dtype=torch.long),
                    torch.full(size=(len(left_id),), fill_value=relation_offset),
                    torch.as_tensor(data=right_id.values, dtype=torch.long),
                ],
                dim=-1,
            )
        )
        # merged factory
        return TriplesFactory(
            mapped_triples=torch.cat(mapped_triples, dim=0),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            **kwargs,
        )


graph_combinator_resolver: ClassResolver[GraphPairCombinator] = ClassResolver.from_subclasses(
    base=GraphPairCombinator,
    default=ExtraRelationGraphCombinator,
)
