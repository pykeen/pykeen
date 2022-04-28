"""Combination strategies for entity alignment datasets."""
import logging
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy
import pandas
import torch
from class_resolver import ClassResolver
from pandas.api.types import is_numeric_dtype, is_string_dtype

from ...triples import CoreTriplesFactory, TriplesFactory
from ...typing import COLUMN_HEAD, COLUMN_TAIL, EA_SIDE_LEFT, EA_SIDE_RIGHT, EA_SIDES, MappedTriples, TargetColumn
from ...utils import format_relative_comparison, get_connected_components

logger = logging.getLogger(__name__)


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


def merge_mappings(
    *pairs: Tuple[str, Mapping[str, int]],
    offsets: torch.LongTensor = None,
    mappings: Sequence[Mapping[int, int]] = None,
    extra: Optional[Mapping[str, int]] = None,
) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for i, (prefix, mapping) in enumerate(pairs):
        for key, value in mapping.items():
            key = f"{prefix}:{key}"
            if offsets:
                value = value + offsets[i].item()
            else:
                assert mappings is not None
                value = mappings[i][value]
            result[key] = value
    if extra:
        result.update(extra)
    return result


def merge_both_mappings(
    left: TriplesFactory,
    right: TriplesFactory,
    relation_offsets: torch.LongTensor,
    # optional
    entity_offsets: Optional[torch.LongTensor] = None,
    extra_relations: Optional[Mapping[str, int]] = None,
    entity_mappings: Sequence[Mapping[int, int]] = None,
) -> Tuple[Mapping[str, int], Mapping[str, int]]:
    # merge entity mapping
    entity_to_id = merge_mappings(
        (EA_SIDE_LEFT, left.entity_to_id),
        (EA_SIDE_RIGHT, right.entity_to_id),
        offsets=entity_offsets,
        mappings=entity_mappings,
    )
    # merge relation mapping
    relation_to_id = merge_mappings(
        (EA_SIDE_LEFT, left.relation_to_id),
        (EA_SIDE_RIGHT, right.relation_to_id),
        offsets=relation_offsets,
        extra=extra_relations,
    )
    return entity_to_id, relation_to_id


def filter_map_alignment(
    alignment: pandas.DataFrame,
    left: CoreTriplesFactory,
    right: CoreTriplesFactory,
    entity_offsets: torch.LongTensor,
) -> torch.LongTensor:
    """
    Convert dataframe with label or ID-based alignment.

    :param alignment: columns: EA_SIDES
        the dataframe with the alignment
    :param left:
        the triples factory of the left graph
    :param right:
        the triples factory of the right graph
    :param entity_offsets: shape: (2,)
        the entity offsets from old to new IDs

    :return: shape: (2, num_alignments)
        the ID-based alignment in new IDs
    """
    # convert labels to IDs
    for side, tf in zip(EA_SIDES, (left, right)):
        if isinstance(tf, TriplesFactory) and is_string_dtype(alignment[side]):
            logger.debug(f"Mapping label-based alignment for {side}")
            # map labels, using -1 as fill-value for invalid labels
            # we cannot drop them here, since the two columns need to stay aligned
            alignment[side] = alignment[side].apply(tf.entity_to_id.get, -1)
        if not is_numeric_dtype(alignment[side]):
            raise ValueError(f"Invalid dype in alignment dataframe for side={side}: {alignment[side].dtype}")

    # filter alignment
    invalid_mask = (alignment < 0).any(axis=1) | alignment.values >= numpy.reshape(
        numpy.asarray([left.num_entities, right.num_entities]), (1, 2)
    )
    if invalid_mask.any():
        logger.warning(
            f"Dropping {format_relative_comparison(part=invalid_mask.sum(), total=alignment.shape[0])} "
            f"alignments due to invalid labels.",
        )
        alignment = alignment.loc[~invalid_mask]

    # map alignment from old IDs to new IDs
    return torch.as_tensor(alignment.to_numpy().T, dtype=torch.long) + entity_offsets.view(1, 2)


def _swap_index(mapped_triples: MappedTriples, dense_map: torch.LongTensor, index: TargetColumn) -> MappedTriples:
    trans = dense_map[mapped_triples[:, index]]
    mask = trans >= 0
    mapped_triples = mapped_triples[mask].clone()
    mapped_triples[:, index] = trans
    return mapped_triples


class GraphPairCombinator:
    """A base class for combination of a graph pair into a single graph."""

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
        # concatenate triples
        mapped_triples, offsets = cat_triples(left, right)
        # filter alignment and translate to IDs
        alignment = filter_map_alignment(alignment=alignment, left=left, right=right, entity_offsets=offsets[:, 0])
        # process
        mapped_triples, alignment, translation_kwargs = self._process(mapped_triples, alignment, offsets)
        if isinstance(left, TriplesFactory) and isinstance(right, TriplesFactory):
            # merge mappings
            entity_to_id, relation_to_id = merge_both_mappings(
                # TODO: offsets can also be dense mapping
                # translation: Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]]
                left=left,
                right=right,
                **translation_kwargs,
            )
            triples_factory = TriplesFactory(
                mapped_triples=mapped_triples,
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
                **kwargs,
            )
        else:
            triples_factory = CoreTriplesFactory(mapped_triples=mapped_triples, **kwargs)

        return triples_factory, alignment

    @abstractmethod
    def _process(
        self,
        mapped_triples: MappedTriples,
        alignment: torch.LongTensor,
        offsets: torch.LongTensor,
    ) -> Tuple[MappedTriples, torch.LongTensor, Mapping[str, Any]]:
        raise NotImplementedError


class DisjointGraphPairCombinator(GraphPairCombinator):
    """This combinator keeps both graphs as disconnected components."""

    # docstr-coverage: inherited
    def _process(
        self,
        mapped_triples: MappedTriples,
        alignment: torch.LongTensor,
        offsets: torch.LongTensor,
    ) -> Tuple[MappedTriples, torch.LongTensor, Mapping[str, Any]]:  # noqa: D102
        return mapped_triples, alignment, dict(offsets=offsets)


class SwapGraphPairCombinator(GraphPairCombinator):
    """Add extra triples by swapping aligned entities."""

    # docstr-coverage: inherited
    def _process(
        self,
        mapped_triples: MappedTriples,
        alignment: torch.LongTensor,
        offsets: torch.LongTensor,
    ) -> Tuple[MappedTriples, torch.LongTensor, Mapping[str, Any]]:  # noqa: D102
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
        return mapped_triples, alignment, dict(offsets=offsets)


class ExtraRelationGraphPairCombinator(GraphPairCombinator):
    """This combinator keeps all entities, but introduces a novel alignment relation."""

    # docstr-coverage: inherited
    def _process(
        self,
        mapped_triples: MappedTriples,
        alignment: torch.LongTensor,
        offsets: torch.LongTensor,
    ) -> Tuple[MappedTriples, torch.LongTensor, Mapping[str, Any]]:  # noqa: D102
        # add alignment triples with extra relation
        left_id, right_id = alignment
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
        return mapped_triples, alignment, dict(offsets=offsets, extra_relations={"same-as": alignment_relation_id})


def _iter_mappings(
    *old_new_ids_pairs: Tuple[torch.LongTensor], offsets: torch.LongTensor
) -> Iterable[Mapping[int, int]]:
    # TODO: check
    old, new = [torch.cat(tensors, dim=0) for tensors in zip(*old_new_ids_pairs)]
    offsets = offsets.tolist()
    for low, high in zip(offsets, offsets[1:]):
        mask = (low <= old) & (old < high)
        this_old = old[mask] - low
        this_new = new[mask]
        yield dict(zip(this_old.tolist(), this_new.tolist()))


class CollapseGraphPairCombinator(GraphPairCombinator):
    """This combinator merges all matching entity pairs into a single ID."""

    # docstr-coverage: inherited
    def _process(
        self,
        mapped_triples: MappedTriples,
        alignment: torch.LongTensor,
        offsets: torch.LongTensor,
    ) -> Tuple[MappedTriples, torch.LongTensor, Mapping[str, Any]]:  # noqa: D102
        # determine connected components regarding the same-as relation (i.e., applies transitivity)
        entity_id_mapping = torch.arange(offsets[-1, 0])
        for cc in get_connected_components(pairs=alignment.tolist()):
            cc = list(cc)
            entity_id_mapping[cc] = min(cc)
        # apply id mapping
        h, r, t = mapped_triples.t()
        h_new, t_new = entity_id_mapping[h], entity_id_mapping[t]
        # ensure consecutive IDs
        inverse = torch.cat([h_new, t_new]).unique(return_inverse=True)[1]
        h_new, t_new = inverse.split(split_size_or_sections=2)
        mapped_triples = torch.stack([h_new, r, t_new], dim=-1)
        # only use training alignments?
        return (
            mapped_triples,
            torch.empty(size=(2, 0), dtype=torch.long),
            dict(entity_mappings=list(_iter_mappings((h, h_new), (t, t_new), offsets=offsets))),
        )


graph_combinator_resolver: ClassResolver[GraphPairCombinator] = ClassResolver.from_subclasses(
    base=GraphPairCombinator,
    default=ExtraRelationGraphPairCombinator,
)
