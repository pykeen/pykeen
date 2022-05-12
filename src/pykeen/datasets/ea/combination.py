"""Combination strategies for entity alignment datasets."""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (
    Any,
    ClassVar,
    DefaultDict,
    Dict,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy
import pandas
import torch
from class_resolver import ClassResolver
from pandas.api.types import is_numeric_dtype, is_string_dtype

from ...triples import CoreTriplesFactory, TriplesFactory
from ...typing import COLUMN_HEAD, COLUMN_TAIL, EA_SIDE_LEFT, EA_SIDE_RIGHT, EA_SIDES, MappedTriples, TargetColumn
from ...utils import format_relative_comparison, get_connected_components

__all__ = [
    # Abstract class
    "GraphPairCombinator",
    # Concrete classes
    "DisjointGraphPairCombinator",
    "SwapGraphPairCombinator",
    "ExtraRelationGraphPairCombinator",
    "CollapseGraphPairCombinator",
    # Data Structures
    "ProcessedTuple",
]

logger = logging.getLogger(__name__)


def cat_shift_triples(*triples: Union[CoreTriplesFactory, MappedTriples]) -> Tuple[MappedTriples, torch.LongTensor]:
    """
    Concatenate (shifted) triples.

    :param triples:
        the triples factories, or mapped triples

    :return:
        a tuple `(combined_triples, offsets)`, where
        * `combined_triples`, shape: `(sum(map(len, triples)), 3)`, is the concatenation of the shifted mapped triples
          such that there is no overlap, and
        * `offsets`, shape: `(len(triples), 2)` comprises the entity & relation offsets for the individual factories
    """
    # a buffer for the triples
    res = []
    # the offsets
    offsets = torch.zeros(len(triples), 2, dtype=torch.long)
    for i, x in enumerate(triples):
        # normalization
        if isinstance(x, CoreTriplesFactory):
            e_offset = x.num_entities
            r_offset = x.num_relations
            x = x.mapped_triples
        else:
            e_offset = x[:, [0, 2]].max().item() + 1
            r_offset = x[:, 1].max().item() + 1
        # append shifted mapped triples
        res.append(x + offsets[None, i, [0, 1, 0]])
        # update offsets
        offsets[i + 1 :, 0] += e_offset
        offsets[i + 1 :, 1] += r_offset
    return torch.cat(res), offsets


def merge_label_to_id_mapping(
    *pairs: Tuple[str, Mapping[str, int]],
    offsets: torch.LongTensor = None,
    mappings: Sequence[Mapping[int, int]] = None,
    extra: Optional[Mapping[str, int]] = None,
) -> Dict[str, int]:
    """
    Merge label-to-id mappings.

    :param pairs:
        pairs of `(prefix, label_to_id)`, where `label_to_id` is the label-to-id mapping, and `prefix` is a string
        prefix to prepend to each key of the label-to-id mapping.
    :param offsets: shape: `(len(pairs),)`
        id offsets for each pair
    :param mappings:
        explicit id remappings for each pair
    :param extra:
        extra entries to add after merging

    :return:
        a merged label-to-id mapping

    :raises ValueError:
        if not exactly one of `offsets` or `mappings` is provided
    """
    if (offsets is None and mappings is None) or (offsets is not None and mappings is not None):
        raise ValueError("Exactly one of `offsets` or `mappings` has to be provided")
    # merge labels with same ID
    value_to_keys: DefaultDict[int, Set[str]] = defaultdict(set)
    for i, (prefix, mapping) in enumerate(pairs):
        for key, value in mapping.items():
            key = f"{prefix}:{key}"
            if offsets is None:
                # for mypy
                assert mappings is not None
                value = mappings[i][value]
            else:
                value = value + offsets[i].item()
            value_to_keys[value].add(key)
    if extra:
        for k, v in extra.items():
            value_to_keys[v].add(k)
    # reconstruct label-to-id
    result: Dict[str, int] = {}
    for value, keys in value_to_keys.items():
        if len(keys) == 1:
            key = list(keys)[0]
        else:
            key = str(set(keys))
        result[key] = value
    return result


def merge_label_to_id_mappings(
    left: TriplesFactory,
    right: TriplesFactory,
    relation_offsets: torch.LongTensor,
    # optional
    entity_offsets: Optional[torch.LongTensor] = None,
    entity_mappings: Sequence[Mapping[int, int]] = None,
    extra_relations: Optional[Mapping[str, int]] = None,
) -> Tuple[Mapping[str, int], Mapping[str, int]]:
    """
    Merge entity-to-id and relation-to-id mappings.

    :param left:
        the left triples factory
    :param right:
        the right triples factory
    :param relation_offsets: shape: (2,)
        the relation offsets
    :param entity_offsets: shape: (2,)
        the entity offsets, if entities are shifted uniformly
    :param entity_mappings:
        explicit entity ID remappings
    :param extra_relations:
        additional relations, as a mapping from their label to IDs

    :return:
        the updated entity and relation to id mappings
    """
    # merge entity mapping
    entity_to_id = merge_label_to_id_mapping(
        (EA_SIDE_LEFT, left.entity_to_id),
        (EA_SIDE_RIGHT, right.entity_to_id),
        offsets=entity_offsets,
        mappings=entity_mappings,
    )
    # merge relation mapping
    relation_to_id = merge_label_to_id_mapping(
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

    :raises ValueError:
        if the datatype of the alignment data frame is imcompatible (neither string nor integer).
    """
    # convert labels to IDs
    for side, tf in zip(EA_SIDES, (left, right)):
        if isinstance(tf, TriplesFactory) and is_string_dtype(alignment[side]):
            logger.debug(f"Mapping label-based alignment for {side}")
            # map labels, using -1 as fill-value for invalid labels
            # we cannot drop them here, since the two columns need to stay aligned
            alignment[side] = alignment[side].apply(tf.entity_to_id.get, args=(-1,))
        if not is_numeric_dtype(alignment[side]):
            raise ValueError(f"Invalid dype in alignment dataframe for side={side}: {alignment[side].dtype}")

    # filter alignment
    invalid_mask = (alignment.values < 0).any(axis=1) | (
        alignment.values >= numpy.reshape(numpy.asarray([left.num_entities, right.num_entities]), newshape=(1, 2))
    ).any(axis=1)
    if invalid_mask.any():
        logger.warning(
            f"Dropping {format_relative_comparison(part=invalid_mask.sum(), total=alignment.shape[0])} "
            f"alignments due to invalid labels.",
        )
        alignment = alignment.loc[~invalid_mask]

    # map alignment from old IDs to new IDs
    return torch.as_tensor(alignment.to_numpy().T, dtype=torch.long) + entity_offsets.view(2, 1)


def swap_index_triples(
    mapped_triples: MappedTriples,
    dense_map: torch.LongTensor,
    index: TargetColumn,
) -> MappedTriples:
    """
    Return triples where some indices in the index column are swapped.

    :param mapped_triples: shape: (m, 3)
        the id-based triples
    :param dense_map: shape: (n,)
        a dense map between IDs. Contains `-1` for missing entries
    :param index:
        the index, a number between 0 (incl) and 3 (excl).

    :return:
        all triples which contain a key at the `index` position, where the key has been replaced by the corresponding
        value in the dense map
    """
    # determine swapping partner
    trans = dense_map[mapped_triples[:, index]]
    # only keep triples where we have a swapping partner
    mask = trans >= 0
    mapped_triples = mapped_triples[mask].clone()
    # replace by swapping partner
    mapped_triples[:, index] = trans[mask]
    return mapped_triples


class ProcessedTuple(NamedTuple):
    """The result of processing a pair of triples factories."""

    #: the merged id-based triples, shape: (n, 3)
    mapped_triples: MappedTriples

    #: the updated alignment, shape: (2, m)
    alignment: torch.LongTensor

    #: additional keyword-based parameters for adjusting label-to-id mappings
    translation_kwargs: Mapping[str, Any]


class GraphPairCombinator(ABC):
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
        mapped_triples, offsets = cat_shift_triples(left, right)
        # filter alignment and translate to IDs
        alignment = filter_map_alignment(alignment=alignment, left=left, right=right, entity_offsets=offsets[:, 0])
        # process
        # TODO: restrict to only using training alignments?
        mapped_triples, alignment, translation_kwargs = self.process(mapped_triples, alignment, offsets)
        if isinstance(left, TriplesFactory) and isinstance(right, TriplesFactory):
            # merge mappings
            entity_to_id, relation_to_id = merge_label_to_id_mappings(
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
            max_ids = mapped_triples.max(axis=0).values
            triples_factory = CoreTriplesFactory(
                mapped_triples=mapped_triples,
                num_entities=max_ids[0::2].max().item(),
                num_relations=max_ids[1].item(),
                **kwargs,
            )

        return triples_factory, alignment

    @abstractmethod
    def process(
        self,
        mapped_triples: MappedTriples,
        alignment: torch.LongTensor,
        offsets: torch.LongTensor,
    ) -> ProcessedTuple:
        """
        Process the combined mapped triples.

        :param mapped_triples: shape: (n, 3)
            the id-based merged triples
        :param alignment: shape: (2, m)
            the id-based entity alignment
        :param offsets: shape: (2, 2)
            the entity and relation offsets from merging

        :return:
            updated triples and alignment tensor, as well as parameters for updating label-to-id mappings
        """
        raise NotImplementedError


class DisjointGraphPairCombinator(GraphPairCombinator):
    """This combinator keeps both graphs as disconnected components."""

    # docstr-coverage: inherited
    def process(
        self,
        mapped_triples: MappedTriples,
        alignment: torch.LongTensor,
        offsets: torch.LongTensor,
    ) -> ProcessedTuple:  # noqa: D102
        return ProcessedTuple(
            mapped_triples,
            alignment,
            dict(entity_offsets=offsets[:, 0], relation_offsets=offsets[:, 1]),
        )


class SwapGraphPairCombinator(GraphPairCombinator):
    """Add extra triples by swapping aligned entities."""

    # docstr-coverage: inherited
    def process(
        self,
        mapped_triples: MappedTriples,
        alignment: torch.LongTensor,
        offsets: torch.LongTensor,
    ) -> ProcessedTuple:  # noqa: D102
        # add swap triples
        # e1 ~ e2 => (e1, r, t) ~> (e2, r, t), or (h, r, e1) ~> (h, r, e2)
        # create dense entity remapping for swap
        dense_map = torch.full(size=(mapped_triples[:, 0::2].max().item() + 1,), fill_value=-1)
        left_id, right_id = alignment
        dense_map[left_id] = right_id
        dense_map[right_id] = left_id
        # add swapped triples
        mapped_triples = torch.cat(
            [
                mapped_triples,
                # swap head
                swap_index_triples(mapped_triples=mapped_triples, dense_map=dense_map, index=COLUMN_HEAD),
                # swap tail
                swap_index_triples(mapped_triples=mapped_triples, dense_map=dense_map, index=COLUMN_TAIL),
            ],
            dim=0,
        )
        return ProcessedTuple(
            mapped_triples,
            alignment,
            dict(entity_offsets=offsets[:, 0], relation_offsets=offsets[:, 1]),
        )


class ExtraRelationGraphPairCombinator(GraphPairCombinator):
    """This combinator keeps all entities, but introduces a novel alignment relation."""

    #: the name of the additional alignment relation
    ALIGNMENT_RELATION_NAME: ClassVar[str] = "same-as"

    # docstr-coverage: inherited
    def process(
        self,
        mapped_triples: MappedTriples,
        alignment: torch.LongTensor,
        offsets: torch.LongTensor,
    ) -> ProcessedTuple:  # noqa: D102
        # add alignment triples with extra relation
        left_id, right_id = alignment
        alignment_relation_id = mapped_triples[:, 1].max().item() + 1
        mapped_triples = torch.cat(
            [
                mapped_triples,
                torch.stack(
                    [
                        left_id,
                        torch.full(size=(len(left_id),), fill_value=alignment_relation_id),
                        right_id,
                    ],
                    dim=-1,
                ),
            ],
            dim=0,
        )
        return ProcessedTuple(
            mapped_triples,
            alignment,
            dict(
                entity_offsets=offsets[:, 0],
                relation_offsets=offsets[:, 1],
                extra_relations={self.ALIGNMENT_RELATION_NAME: alignment_relation_id},
            ),
        )


def iter_entity_mappings(
    *old_new_ids_pairs: Tuple[torch.LongTensor, torch.LongTensor], offsets: torch.LongTensor
) -> Iterable[Mapping[int, int]]:
    """
    Create explicit Id mappings.

    :param old_new_ids_pairs:
        aligned pairs of old and new ids
    :param offsets: shape: (2,)
        the entity offsets

    :yields: explicit id remappings
    """
    old, new = [torch.cat(tensors, dim=0) for tensors in zip(*old_new_ids_pairs)]
    offsets = offsets.tolist() + [old.max().item() + 1]
    for low, high in zip(offsets, offsets[1:]):
        mask = (low <= old) & (old < high)
        this_old = old[mask] - low
        this_new = new[mask]
        yield dict(zip(this_old.tolist(), this_new.tolist()))


class CollapseGraphPairCombinator(GraphPairCombinator):
    """This combinator merges all matching entity pairs into a single ID."""

    # docstr-coverage: inherited
    def process(
        self,
        mapped_triples: MappedTriples,
        alignment: torch.LongTensor,
        offsets: torch.LongTensor,
    ) -> ProcessedTuple:  # noqa: D102
        # determine connected components regarding the same-as relation (i.e., applies transitivity)
        entity_id_mapping = torch.arange(mapped_triples[:, 0::2].max().item() + 1)
        for cc in get_connected_components(pairs=alignment.t().tolist()):
            cc = list(cc)
            entity_id_mapping[cc] = min(cc)
        # apply id mapping
        h, r, t = mapped_triples.t()
        h_new, t_new = entity_id_mapping[h], entity_id_mapping[t]
        # ensure consecutive IDs
        inverse = torch.cat([h_new, t_new]).unique(return_inverse=True)[1]
        h_new, t_new = inverse.split((len(h), len(t)))
        mapped_triples = torch.stack([h_new, r, t_new], dim=-1)
        # only use training alignments?
        return ProcessedTuple(
            mapped_triples,
            torch.empty(size=(2, 0), dtype=torch.long),
            dict(
                entity_mappings=list(iter_entity_mappings((h, h_new), (t, t_new), offsets=offsets[:, 0])),
                relation_offsets=offsets[:, 1],
            ),
        )


graph_combinator_resolver: ClassResolver[GraphPairCombinator] = ClassResolver.from_subclasses(
    base=GraphPairCombinator,
    default=ExtraRelationGraphPairCombinator,
)
