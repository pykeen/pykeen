"""Combination strategies for entity alignment datasets."""
import logging
from abc import abstractmethod
from typing import Dict, Mapping, Tuple, Union

import pandas
import torch
from class_resolver import ClassResolver

from ...triples import CoreTriplesFactory, TriplesFactory
from ...typing import COLUMN_HEAD, COLUMN_TAIL, EA_SIDE_LEFT, EA_SIDE_RIGHT, MappedTriples, TargetColumn
from ...utils import format_relative_comparison, get_connected_components

logger = logging.getLogger(__name__)


# TODO: support ID-only graphs


def cat_triples(*triples: Union[CoreTriplesFactory, MappedTriples]) -> Tuple[MappedTriples, torch.LongTensor]:
    """
    Concatenate (shifted) triples.

    :param tfs:
        the triples factories, or mapped triples

    :return:
        the concatenation of the shifted mapped triples such that there is no overlap, and the offsets for the
        individual factories
    """
    # a buffer for the triples
    res = []
    # the overall offsets
    offsets = torch.empty(len(triples), 3)
    # the current offset
    offset = torch.zeros(1, 3, dtype=torch.long)
    for i, x in enumerate(triples):
        # store offset
        offsets[i] = offset
        # normalization
        if isinstance(x, CoreTriplesFactory):
            x = x.mapped_triples
            e_offset = x.num_entities
            r_offset = x.num_relations
        else:
            e_offset = x[:, [0, 2]].max().item() + 1
            r_offset = x[:, 1].max().item() + 1
        # append shifted mapped triples
        res.append(x + offset)
        # update offset
        offset[[0, 2]] += e_offset
        offset[1] += r_offset
    return torch.cat(res), offsets


def merge_mappings(*pairs: Tuple[str, Mapping[str, int]], offsets: torch.LongTensor) -> Mapping[str, int]:
    result: Dict[str, int] = {}
    for (prefix, mapping), offset in zip(pairs, offsets.tolist()):
        result.update((f"{prefix}:{key}", value + offset) for key, value in mapping.items())
    return result


def merge_both_mappings(
    left: TriplesFactory, right: TriplesFactory, offsets: torch.LongTensor
) -> Tuple[Mapping[str, int], Mapping[str, int]]:
    return (
        merge_mappings(
            (EA_SIDE_LEFT, left.entity_to_id),
            (EA_SIDE_RIGHT, right.entity_to_id),
            offsets=offsets[:, 0],
        ),
        merge_mappings(
            (EA_SIDE_LEFT, left.relation_to_id),
            (EA_SIDE_RIGHT, right.relation_to_id),
            offsets=offsets[:, 1],
        ),
    )


def filter_map_alignment(
    alignment: pandas.DataFrame,
    left: TriplesFactory,
    right: TriplesFactory,
    entity_offsets: torch.LongTensor,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    # filter alignment
    mask = ~(alignment[EA_SIDE_LEFT].isin(left.entity_to_id) & alignment[EA_SIDE_RIGHT].isin(right.entity_to_id))
    if mask.any():
        alignment = alignment.loc[~mask]
        logger.warning(
            f"Dropped {format_relative_comparison(part=mask.sum(), total=alignment.shape[0])} "
            f"alignments due to unknown labels.",
        )

    # map alignment to (new) IDs
    left_id = alignment[EA_SIDE_LEFT].apply(left.entity_to_id.__getitem__) + entity_offsets[0]
    right_id = alignment[EA_SIDE_RIGHT].apply(right.entity_to_id.__getitem__) + entity_offsets[1]

    return left_id, right_id


class GraphPairCombinator:
    """A base class for combination of a graph pair into a single graph."""

    @abstractmethod
    def __call__(
        self,
        left: TriplesFactory,
        right: TriplesFactory,
        alignment: pandas.DataFrame,
        **kwargs,
    ) -> Tuple[TriplesFactory, torch.LongTensor]:
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
            a single triples factory comprising the joint graph, as well as a tensor of pairs of matching IDs.
            The tensor of matching pairs has shape `(2, num_alignments)`, where `num_alignments` can also be 0.
        """
        raise NotImplementedError


class DisjointGraphPairCombinator(GraphPairCombinator):
    """This combinator keeps both graphs as disconnected components."""

    def __call__(
        self, left: TriplesFactory, right: TriplesFactory, alignment: pandas.DataFrame, **kwargs
    ) -> Tuple[TriplesFactory, torch.LongTensor]:
        # concatenate triples
        mapped_triples, offsets = cat_triples(left, right)
        # merge mappings
        entity_to_id, relation_to_id = merge_both_mappings(left=left, right=right, offsets=offsets)
        # filter alignment and translate to IDs
        alignment = torch.stack(
            filter_map_alignment(alignment=alignment, left=left, right=right, entity_offsets=offsets[:, 0]),
            dim=0,
        )
        return (
            TriplesFactory(
                mapped_triples=mapped_triples,
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
                **kwargs,
            ),
            alignment,
        )


def _swap_index(mapped_triples: MappedTriples, dense_map: torch.LongTensor, index: TargetColumn) -> MappedTriples:
    trans = dense_map[mapped_triples[:, index]]
    mask = trans >= 0
    mapped_triples = mapped_triples[mask].clone()
    mapped_triples[:, index] = trans
    return mapped_triples


class SwapGraphPairCombinator(GraphPairCombinator):
    """Add extra triples by swapping aligned entities."""

    def __call__(
        self, left: TriplesFactory, right: TriplesFactory, alignment: pandas.DataFrame, **kwargs
    ) -> Tuple[TriplesFactory, torch.LongTensor]:
        # concatenate triples
        mapped_triples, offsets = cat_triples(left, right)
        # merge mappings
        entity_to_id, relation_to_id = merge_both_mappings(left=left, right=right, offsets=offsets)
        # filter alignment and translate to IDs
        alignment = torch.stack(
            filter_map_alignment(alignment=alignment, left=left, right=right, entity_offsets=offsets[:, 0]),
            dim=0,
        )
        # add swap triples
        # e1 ~ e2 => (e1, r, t) ~> (e2, r, t), or (h, r, e1) ~> (h, r, e2)
        dense_map = torch.full(size=(offsets[-1, 0],), fill_value=-1)
        left_id, right_id = alignment
        dense_map[left_id] = right_id
        dense_map[right_id] = left_id
        mapped_triples = torch.cat(
            [
                mapped_triples,
                # swap head
                _swap_index(mapped_triples=mapped_triples, dense_map=dense_map, index=COLUMN_HEAD),
                # swap tail
                _swap_index(mapped_triples=mapped_triples, dense_map=dense_map, index=COLUMN_TAIL),
            ],
            dim=0,
        )

        return (
            TriplesFactory(
                mapped_triples=mapped_triples,
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
                **kwargs,
            ),
            alignment,
        )


class ExtraRelationGraphPairCombinator(GraphPairCombinator):
    """This combinator keeps all entities, but introduces a novel alignment relation."""

    def __call__(
        self,
        left: TriplesFactory,
        right: TriplesFactory,
        alignment: pandas.DataFrame,
        **kwargs,
    ) -> TriplesFactory:  # noqa: D102
        # concatenate triples
        mapped_triples, offsets = cat_triples(left, right)

        # filter alignment and translate to IDs
        left_id, right_id = filter_map_alignment(
            alignment=alignment, left=left, right=right, entity_offsets=offsets[:, 0]
        )
        # add alignment triples with extra relation
        alignment_relation_id = offsets[-1, 1]
        mapped_triples.append(
            torch.stack(
                [
                    torch.as_tensor(data=left_id.values, dtype=torch.long),
                    torch.full(size=(len(left_id),), fill_value=alignment_relation_id),
                    torch.as_tensor(data=right_id.values, dtype=torch.long),
                ],
                dim=-1,
            )
        )

        # merge mappings
        entity_to_id, relation_to_id = merge_both_mappings(left=left, right=right, offsets=offsets)
        # extra alignment relation
        relation_to_id["same-as"] = alignment_relation_id = offsets[-1, 1]

        # merged factory
        return TriplesFactory(
            mapped_triples=torch.cat(mapped_triples, dim=0),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            **kwargs,
        ), torch.stack([left_id, right_id], dim=0)


class CollapseGraphPairCombinator(GraphPairCombinator):
    """This combinator merges all matching entity pairs into a single ID."""

    def __call__(
        self,
        left: TriplesFactory,
        right: TriplesFactory,
        alignment: pandas.DataFrame,
        **kwargs,
    ) -> TriplesFactory:  # noqa: D102
        mapped_triples, offsets = cat_triples(left, right)
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
        ), torch.empty(size=(2, 0), dtype=torch.long)


graph_combinator_resolver: ClassResolver[GraphPairCombinator] = ClassResolver.from_subclasses(
    base=GraphPairCombinator,
    default=ExtraRelationGraphPairCombinator,
)
