# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

import dataclasses
import logging
import os
import re
from collections import Counter, defaultdict
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
from tqdm.autonotebook import tqdm

from .instances import LCWAInstances, SLCWAInstances
from .utils import load_triples
from ..typing import EntityMapping, LabeledTriples, MappedTriples, RelationMapping
from ..utils import compact_mapping, invert_mapping, random_non_negative_int, slice_triples

__all__ = [
    'TriplesFactory',
    'create_entity_mapping',
    'create_relation_mapping',
    'INVERSE_SUFFIX',
]

logger = logging.getLogger(__name__)

INVERSE_SUFFIX = '_inverse'
TRIPLES_DF_COLUMNS = ('head_id', 'head_label', 'relation_id', 'relation_label', 'tail_id', 'tail_label')


def get_unique_entity_ids_from_triples_tensor(mapped_triples: MappedTriples) -> torch.LongTensor:
    """Return the unique entity IDs used in a tensor of triples."""
    return mapped_triples[:, [0, 2]].unique()


def _create_multi_label_tails_instance(
    mapped_triples: MappedTriples,
    use_tqdm: Optional[bool] = None,
) -> Dict[Tuple[int, int], List[int]]:
    """Create for each (h,r) pair the multi tail label."""
    logger.debug('Creating multi label tails instance')

    '''
    The mapped triples matrix has to be a numpy array to ensure correct pair hashing, as explained in
    https://github.com/pykeen/pykeen/commit/1bc71fe4eb2f24190425b0a4d0b9d6c7b9c4653a
    '''
    mapped_triples = mapped_triples.cpu().detach().numpy()

    s_p_to_multi_tails_new = _create_multi_label_instances(
        mapped_triples,
        element_1_index=0,
        element_2_index=1,
        label_index=2,
        use_tqdm=use_tqdm,
    )

    logger.debug('Created multi label tails instance')

    return s_p_to_multi_tails_new


def _create_multi_label_instances(
    mapped_triples: MappedTriples,
    element_1_index: int,
    element_2_index: int,
    label_index: int,
    use_tqdm: Optional[bool] = None,
) -> Dict[Tuple[int, int], List[int]]:
    """Create for each (element_1, element_2) pair the multi-label."""
    instance_to_multi_label = defaultdict(set)

    if use_tqdm is None:
        use_tqdm = True

    it = mapped_triples
    if use_tqdm:
        it = tqdm(mapped_triples, unit='triple', unit_scale=True, desc='Grouping triples')
    for row in it:
        instance_to_multi_label[row[element_1_index], row[element_2_index]].add(row[label_index])

    # Create lists out of sets for proper numpy indexing when loading the labels
    # TODO is there a need to have a canonical sort order here?
    instance_to_multi_label_new = {
        key: list(value)
        for key, value in instance_to_multi_label.items()
    }

    return instance_to_multi_label_new


def create_mapping(
    labels: Union[np.ndarray, Collection[str]],
    sort_key: Optional[Callable[[str], Any]] = None,
) -> Mapping[str, int]:
    """Create a mapping from unique labels to consecutive IDs from a given collection of labels."""
    if isinstance(labels, np.ndarray):
        labels = np.unique(labels).tolist()

    # Sorting ensures consistent results when the input is permuted
    labels = sorted(set(labels), key=sort_key)

    return {
        str(label): id_
        for id_, label in enumerate(labels)
    }


def create_entity_mapping(triples: LabeledTriples) -> EntityMapping:
    """Create mapping from entity labels to IDs.

    :param triples: shape: (n, 3), dtype: str
    """
    return create_mapping(triples[:, [0, 2]])


def _relation_sort_key(relation_label: str) -> Tuple[str, bool]:
    """Return the sort key for relation-labels collating inverse relations."""
    return (
        re.sub(f'{INVERSE_SUFFIX}$', '', relation_label),
        relation_label.endswith(f'{INVERSE_SUFFIX}'),
    )


def create_relation_mapping(relations: Union[np.ndarray, Collection[str]]) -> RelationMapping:
    """Create mapping from relation labels to IDs."""
    return create_mapping(labels=relations, sort_key=_relation_sort_key)


def _map_triples_elements_to_ids(
    triples: LabeledTriples,
    entity_to_id: EntityMapping,
    relation_to_id: RelationMapping,
) -> MappedTriples:
    """Map entities and relations to pre-defined ids."""
    if triples.size == 0:
        logger.warning('Provided empty triples to map.')
        return torch.empty(0, 3, dtype=torch.long)

    heads, relations, tails = slice_triples(triples)

    # When triples that don't exist are trying to be mapped, they get the id "-1"
    entity_getter = np.vectorize(entity_to_id.get)
    head_column = entity_getter(heads, [-1])
    tail_column = entity_getter(tails, [-1])
    relation_getter = np.vectorize(relation_to_id.get)
    relation_column = relation_getter(relations, [-1])

    # Filter all non-existent triples
    head_filter = head_column < 0
    relation_filter = relation_column < 0
    tail_filter = tail_column < 0
    num_no_head = head_filter.sum()
    num_no_relation = relation_filter.sum()
    num_no_tail = tail_filter.sum()

    if (num_no_head > 0) or (num_no_relation > 0) or (num_no_tail > 0):
        logger.warning(
            f"You're trying to map triples with {num_no_head + num_no_tail} entities and {num_no_relation} relations"
            f" that are not in the training set. These triples will be excluded from the mapping.",
        )
        non_mappable_triples = (head_filter | relation_filter | tail_filter)
        head_column = head_column[~non_mappable_triples, None]
        relation_column = relation_column[~non_mappable_triples, None]
        tail_column = tail_column[~non_mappable_triples, None]
        logger.warning(
            f"In total {non_mappable_triples.sum():.0f} from {triples.shape[0]:.0f} triples were filtered out",
        )

    triples_of_ids = np.concatenate([head_column, relation_column, tail_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.long)
    # Note: Unique changes the order of the triples
    # Note: Using unique means implicit balancing of training samples
    unique_mapped_triples = np.unique(ar=triples_of_ids, axis=0)
    return torch.tensor(unique_mapped_triples, dtype=torch.long)


def _check_already_inverted_relations(relations: Iterable[str]) -> bool:
    return any(relation.endswith(INVERSE_SUFFIX) for relation in relations)


def _map_column(
    labeled_column: Union[np.ndarray, Sequence[str], str],
    label_to_id: Mapping[str, int],
    name: str,
    vectorized_mapper: Callable[[np.ndarray, Tuple[int]], np.ndarray],
    unknown_id: int = -1,
    raise_on_unknown: bool = True,
) -> torch.LongTensor:
    """Apply a vectorized mapping to a column of labels.

    :param labeled_column:
        The column of labels.
    :param label_to_id:
        The mapping from label to ID.
    :param name:
        The name used for log and error messages.
    :param vectorized_mapper:
        The vectorized label-to-id mapper.
    :param unknown_id:
        An ID to use for unknown labels.
    :param raise_on_unknown:
        Whether to raise a ValueError when encountering unknown labels.

    :return:
        An array of IDs.
    """
    # Input normalization
    if isinstance(labeled_column, str):
        labeled_column = [labeled_column]
    labeled_column = np.asanyarray(labeled_column)

    if unknown_id is None:
        unknown_id = min(label_to_id.values()) - 1
        logger.info(f'Inferred unknown_id={unknown_id} for {name} ID.')
    if unknown_id in label_to_id.values():
        raise ValueError(f'unknown_id={unknown_id} is used as {name} ID!')

    mapped_column = vectorized_mapper(labeled_column, (unknown_id,))

    # check for unknown
    if raise_on_unknown and (unknown_id == mapped_column).any():
        raise ValueError(f'Input contains unknown {name}!')

    # convert to tensor
    return torch.as_tensor(data=mapped_column, dtype=torch.long)


def _label_column(
    mapped_column: Union[np.ndarray, Sequence[int], torch.LongTensor, int],
    vectorized_labeler: Callable[[np.ndarray, Tuple[str]], np.ndarray],
    unknown_label: str,
) -> np.ndarray:
    """Apply vectorized labeler to an array of IDs.

    :param mapped_column:
        The IDs.
    :param vectorized_labeler:
        The vectorized labeler.
    :param unknown_label:
        A label to use for unknown IDs.

    :return:
        An array of labels, same shape as mapped_column.
    """
    # Input normalization
    if isinstance(mapped_column, int):
        mapped_column = [mapped_column]
    if torch.is_tensor(mapped_column):
        mapped_column = mapped_column.cpu().numpy()
    mapped_column = np.asanyarray(mapped_column)

    # Actual labeling
    return vectorized_labeler(mapped_column, (unknown_label,))


@dataclasses.dataclass
class Triples:
    """ID-based triples."""
    # TODO: Replace usage of MappedTriples by this class

    #: The mapped triples, shape: (n, 3)
    mapped_triples: torch.LongTensor

    #: The maximum entity ID
    max_entity_id: int = None

    #: The maximum relation ID
    max_relation_id: int = None

    #: Whether inverse triples are contained
    contains_inverse_triples: bool = False

    def __post_init__(self):
        # Infer maximum entity ID from triples if not given
        max_entity_id = self.mapped_triples[:, [0, 2]].max()
        if self.max_entity_id is None:
            self.max_entity_id = max_entity_id + 1
            logger.info(f'Inferred maximum entity ID from triples: maximum_entity_id={self.max_entity_id}.')

        # Infer maximum relation ID from triples if not given
        max_relation_id = self.mapped_triples[:, 1].max()
        if self.max_relation_id is None:
            self.max_relation_id = max_relation_id + 1
            logger.info(f'Inferred maximum relation ID from triples: maximum_relation_id={self.max_relation_id}.')

        # Maximum entity ID validation
        if max_entity_id > self.max_entity_id:
            raise ValueError(
                f'Maximum entity ID present in triples ({max_entity_id}) is not less than the one provided '
                f'({self.max_entity_id}).'
            )

        # Maximum relation ID validation
        if not self.contains_inverse_triples:
            if max_relation_id >= self.max_relation_id:
                raise ValueError(
                    f'Maximum relation ID present in triples ({max_relation_id}) is not less than the one provided '
                    f'({self.max_relation_id}).'
                )
        else:
            if max_relation_id >= 2 * self.max_relation_id:
                # TODO:
                raise ValueError('...')
            elif max_relation_id == self.max_relation_id - 1:
                logger.warning('contains_inverse_triples=True, but the triples look like they do not contain such.')

    def with_inverse_triples(self) -> 'Triples':
        """Add inverse triples, or return instance if already present."""
        if self.contains_inverse_triples:
            logger.warning('Triples object contains already inverse triples.')
            return self

        triples = self.mapped_triples
        inverse_triples = triples.clone().flip(-1)
        # TODO: +1 variant
        inverse_triples[:, 1] += self.max_relation_id
        return Triples(
            mapped_triples=torch.cat([
                triples,
                inverse_triples,
            ], dim=0),
            max_entity_id=self.max_entity_id,
            max_relation_id=self.max_relation_id,
            contains_inverse_triples=True,
        )


@dataclasses.dataclass
class LabelMapping:
    """
    A mapping between labels to IDs.

    Within this class we use the following terminology:

    * mapping:
        A conversion from labels to IDs.
    * labeling:
        A conversion from IDs to labels.
    """

    #: The mapping for entities
    entity_label_to_id: EntityMapping

    #: The mapping for relations
    relation_label_to_id: RelationMapping

    #: The inverse mapping for entity_label_to_id; initialized automatically
    entity_id_to_label: Mapping[int, str] = dataclasses.Field(init=False)

    #: The inverse mapping for relation_label_to_id; initialized automatically
    relation_id_to_label: Mapping[int, str] = dataclasses.Field(init=False)

    #: A vectorized version of entity_label_to_id; initialized automatically
    _vectorized_entity_mapper: Callable[[np.ndarray, Tuple[int]], np.ndarray] = dataclasses.Field(init=False)

    #: A vectorized version of relation_label_to_id; initialized automatically
    _vectorized_relation_mapper: Callable[[np.ndarray, Tuple[int]], np.ndarray] = dataclasses.Field(init=False)

    #: A vectorized version of entity_id_to_label; initialized automatically
    _vectorized_entity_labeler: Callable[[np.ndarray, Tuple[str]], np.ndarray] = dataclasses.Field(init=False)

    #: A vectorized version of relation_id_to_label; initialized automatically
    _vectorized_relation_labeler: Callable[[np.ndarray, Tuple[str]], np.ndarray] = dataclasses.Field(init=False)

    @staticmethod
    def from_labeled_triples(triples: LabeledTriples) -> 'LabelMapping':
        """Create a mapping from labeled triples."""
        return LabelMapping(
            entity_label_to_id=create_entity_mapping(triples=triples),
            relation_label_to_id=create_relation_mapping(relations=triples[:, 1]),
        )

    def __post_init__(self):
        """Pre-compute derived mappings."""
        # ID to label mapping
        self.entity_id_to_label = invert_mapping(mapping=self.entity_label_to_id)
        self.relation_id_to_label = invert_mapping(mapping=self.relation_label_to_id)

        # vectorized versions
        self._vectorized_entity_mapper = np.vectorize(self.entity_label_to_id.get)
        self._vectorized_relation_mapper = np.vectorize(self.relation_label_to_id.get)
        self._vectorized_entity_labeler = np.vectorize(self.entity_id_to_label.get)
        self._vectorized_relation_labeler = np.vectorize(self.relation_id_to_label.get)

    @property
    def max_entity_id(self) -> int:
        """Return the maximum entity ID plus 1."""
        return max(self.entity_label_to_id.values()) + 1

    @property
    def max_relation_id(self) -> int:
        """Return the maximum relation ID plus 1."""
        return max(self.relation_label_to_id.values()) + 1

    @property
    def is_compact(self) -> bool:
        """Whether the mapping is compact, i.e. the IDs are consecutive from 0 to num_choices - 1."""
        return all(
            set(mapping.values()) == set(range(len(mapping)))
            for mapping in (self.entity_label_to_id, self.relation_label_to_id)
        )

    def compact(self) -> 'LabelMapping':
        """Return a compact version of the label mapping."""
        # No need for compaction
        if self.is_compact:
            logger.debug('Label mapping is already compact.')
            return self

        # TODO: Return compaction, i.e. the mapping from old to new ID?
        return LabelMapping(
            entity_label_to_id=compact_mapping(mapping=self.entity_label_to_id)[0],
            relation_label_to_id=compact_mapping(mapping=self.relation_label_to_id)[0],
        )

    def map_entities(
        self,
        entities: Union[np.ndarray, Sequence[str], str],
        unknown_id: int = -1,
        raise_on_unknown: bool = True,
    ) -> torch.LongTensor:
        """Convert entity labels to the corresponding IDs."""
        return _map_column(
            labeled_column=entities,
            label_to_id=self.entity_label_to_id,
            name='entity',
            vectorized_mapper=self._vectorized_entity_mapper,
            unknown_id=unknown_id,
            raise_on_unknown=raise_on_unknown,
        )

    def map_relations(
        self,
        relations: Union[np.ndarray, Sequence[str], str],
        unknown_id: int = -1,
        raise_on_unknown: bool = True,
    ) -> torch.LongTensor:
        """Convert entity labels to the corresponding IDs."""
        return _map_column(
            labeled_column=relations,
            label_to_id=self.relation_label_to_id,
            name='relation',
            vectorized_mapper=self._vectorized_relation_mapper,
            unknown_id=unknown_id,
            raise_on_unknown=raise_on_unknown,
        )

    def map_triples(
        self,
        triples: Union[LabeledTriples, Sequence[Tuple[str, str, str]], Tuple[str, str, str]],
        drop_duplicates: bool = True,
        raise_on_unknown: bool = False,
        drop_unknown: bool = True,
        unknown_id: Optional[int] = -1,
    ) -> Triples:
        """Convert label-based triples to ID-based triples."""
        # Input normalization
        triples = np.asanyarray(triples)
        triples = np.atleast_2d(triples)

        # vectorized mapping
        mapped_triples = torch.as_tensor(
            data=np.stack(
                [
                    self.map_entities(
                        entities=triples[:, 0],
                        unknown_id=unknown_id,
                        raise_on_unknown=raise_on_unknown,
                    ),
                    self.map_relations(
                        relations=triples[:, 1],
                        unknown_id=unknown_id,
                        raise_on_unknown=raise_on_unknown,
                    ),
                    self.map_entities(
                        entities=triples[:, 2],
                        unknown_id=unknown_id,
                        raise_on_unknown=raise_on_unknown,
                    ),
                ],
                axis=-1,
            ),
            dtype=torch.long,
        )

        # drop unknowns
        unknown: torch.BoolTensor = (mapped_triples < 0)
        num_unknown_entities = unknown[[0, 2]].sum()
        num_unknown_relations = unknown[1]
        if drop_unknown:
            if num_unknown_entities > 0 or num_unknown_relations > 0:
                drop_mask = unknown.any(dim=-1)
                num_dropped_triples = drop_mask.sum()
                logger.warning(
                    f"You're trying to map triples with {num_unknown_entities} entities and {num_unknown_relations} "
                    f"relations that are not in the training set. These triples will be excluded from the mapping. In "
                    f"total {num_dropped_triples}/{triples.shape[0]} triples are dropped."
                )
                mapped_triples = mapped_triples[~drop_mask, :]
        else:
            num_triples_with_unknown = unknown.sum(axis=-1)
            logger.warning(f'Keeping {num_triples_with_unknown} triples with at least one unknown entity/relation.')

        # drop duplicates
        if drop_duplicates:
            old_num_triples = mapped_triples.shape[0]

            # Comment: The underlying implementation might still sort the array.
            mapped_triples = mapped_triples.unique(sorted=False, dim=0)

            new_num_triples = mapped_triples.shape[0]
            if new_num_triples < old_num_triples:
                logger.warning(
                    f'Dropped {old_num_triples - new_num_triples}/{old_num_triples} triples due to being duplicates.'
                )

        return Triples(
            mapped_triples=mapped_triples,
            max_entity_id=self.max_entity_id,
            max_relation_id=self.max_relation_id,
            contains_inverse_triples=False,
        )

    def label_entities(
        self,
        mapped_entities: Union[np.ndarray, Sequence[int], torch.LongTensor, int],
        unknown_label: Optional[str] = None,
    ) -> np.ndarray:
        """Convert entity IDs to labels."""
        return _label_column(
            mapped_column=mapped_entities,
            vectorized_labeler=self._vectorized_entity_labeler,
            unknown_label=unknown_label or 'UNKNOWN_ENTITY',
        )

    def label_relations(
        self,
        mapped_relations: Union[np.ndarray, Sequence[int], torch.LongTensor, int],
        unknown_label: Optional[str] = None,
    ) -> np.ndarray:
        """Convert relation IDs to labels."""
        return _label_column(
            mapped_column=mapped_relations,
            vectorized_labeler=self._vectorized_entity_labeler,
            unknown_label=unknown_label or 'UNKNOWN_RELATION',
        )

    def label_triples(
        self,
        mapped_triples: MappedTriples,
        unknown_entity_label: Optional[str] = None,
        unknown_relation_label: Optional[str] = None,
    ) -> LabeledTriples:
        """Convert ID-based triples to label-based triples."""
        if not torch.is_tensor(mapped_triples):
            mapped_triples = np.asanyarray(mapped_triples)
        return np.stack([
            self.label_entities(mapped_entities=mapped_triples[:, 0], unknown_label=unknown_entity_label),
            self.label_relations(mapped_relations=mapped_triples[:, 1], unknown_label=unknown_relation_label),
            self.label_entities(mapped_entities=mapped_triples[:, 2], unknown_label=unknown_entity_label),
        ], axis=-1)


class TriplesFactory:
    """Create instances given the path to triples."""

    #: The mapping from entities' labels to their indices
    entity_to_id: EntityMapping

    #: The mapping from relations' labels to their indices
    relation_to_id: RelationMapping

    #: A three-column matrix where each row are the head label,
    #: relation label, then tail label
    triples: LabeledTriples

    #: A three-column matrix where each row are the head identifier,
    #: relation identifier, then tail identifier
    mapped_triples: MappedTriples

    #: A dictionary mapping each relation to its inverse, if inverse triples were created
    relation_to_inverse: Optional[Mapping[str, str]]

    def __init__(
        self,
        *,
        triples: LabeledTriples,
        mapped_triples: MappedTriples,
        entity_to_id: EntityMapping,
        relation_to_id: RelationMapping,
        relation_to_inverse: Optional[Mapping[str, str]],
    ) -> None:
        """Initialize the triples factory."""
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.triples = triples
        self.mapped_triples = mapped_triples
        self.relation_to_inverse = relation_to_inverse

    @staticmethod
    def from_path(
        path: Union[str, TextIO] = None,
        create_inverse_triples: bool = False,
        entity_to_id: Optional[EntityMapping] = None,
        relation_to_id: Optional[RelationMapping] = None,
        compact_id: bool = True,
    ) -> 'TriplesFactory':
        """
        Instantiate triples factory from a TSV file.

        :param path:
            The path to a 3-column TSV file with triples in it.
        :param entity_to_id:
            If provided use this mapping to translate from entity labels to IDs.
        :param relation_to_id:
            If provided use this mapping to translate from relation labels to IDs.
        :param create_inverse_triples:
            Should inverse triples be created?
        :param compact_id:
            Whether to compact the IDs such that they range from 0 to (num_entities or num_relations)-1.
        """
        if isinstance(path, str):
            path = os.path.abspath(path)
        elif isinstance(path, TextIO):
            path = os.path.abspath(path.name)
        else:
            raise TypeError(f'path is invalid type: {type(path)}')

        # TODO: Check if lazy evaluation would make sense
        triples = load_triples(path)

        return TriplesFactory.from_triples(
            triples=triples,
            create_inverse_triples=create_inverse_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            compact_id=compact_id,
        )

    @staticmethod
    def from_triples(
        triples: LabeledTriples,
        create_inverse_triples: bool = False,
        entity_to_id: Optional[EntityMapping] = None,
        relation_to_id: Optional[RelationMapping] = None,
        compact_id: bool = True,
    ) -> 'TriplesFactory':
        """
        Instantiate triples factory from an array of labeled triples.

        :param triples:
            The triples, a (n, 3) array of triples in label format.
        :param entity_to_id:
            If provided use this mapping to translate from entity labels to IDs.
        :param relation_to_id:
            If provided use this mapping to translate from relation labels to IDs.
        :param create_inverse_triples:
            Should inverse triples be created?
        :param compact_id:
            Whether to compact the IDs such that they range from 0 to (num_entities or num_relations)-1.
        """
        # TODO: Inverse relations
        relations = triples[:, 1]
        unique_relations = set(relations)

        # Check if the triples are inverted already
        relations_already_inverted = _check_already_inverted_relations(unique_relations)

        if create_inverse_triples or relations_already_inverted:
            if relations_already_inverted:
                logger.info(
                    f'Some triples already have suffix {INVERSE_SUFFIX}. '
                    f'Creating TriplesFactory based on inverse triples',
                )
                relation_to_inverse = {
                    re.sub('_inverse$', '', relation): f"{re.sub('_inverse$', '', relation)}{INVERSE_SUFFIX}"
                    for relation in unique_relations
                }

            else:
                relation_to_inverse = {
                    relation: f"{relation}{INVERSE_SUFFIX}"
                    for relation in unique_relations
                }
                inverse_triples = np.stack(
                    [
                        triples[:, 2],
                        np.array([relation_to_inverse[relation] for relation in relations], dtype=np.str),
                        triples[:, 0],
                    ],
                    axis=-1,
                )
                # extend original triples with inverse ones
                triples = np.concatenate([triples, inverse_triples], axis=0)
                _num_relations = 2 * len(unique_relations)

        else:
            relation_to_inverse = None

        # Generate entity mapping if necessary
        if entity_to_id is None:
            entity_to_id = create_entity_mapping(triples=triples)
        if compact_id:
            entity_to_id = compact_mapping(mapping=entity_to_id)[0]

        # Generate relation mapping if necessary
        if relation_to_id is None:
            relation_to_id = create_relation_mapping(relations=triples[:, 1])
        if compact_id:
            relation_to_id = compact_mapping(mapping=relation_to_id)[0]

        # Map triples of labels to triples of IDs.
        mapped_triples = _map_triples_elements_to_ids(
            triples=triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        return TriplesFactory(
            triples=triples,
            mapped_triples=mapped_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            relation_to_inverse=relation_to_inverse,
        )

    @staticmethod
    def from_mapped_triples(
        mapped_triples: MappedTriples,
        entity_to_id: EntityMapping,
        relation_to_id: RelationMapping,
        relation_to_inverse: Optional[Mapping[str, str]] = None,
    ) -> 'TriplesFactory':
        """Create a triples factory from ID-based triples."""
        # translate to labeled triples, since the triples factory needs also the labeled triples
        triples = np.asanyarray(
            [entity_to_id[h], relation_to_id[r], entity_to_id[t]]
            for h, r, t in mapped_triples.numpy()
        )

        return TriplesFactory(
            triples=triples,
            mapped_triples=mapped_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            relation_to_inverse=relation_to_inverse,
        )

    @property
    def num_entities(self) -> int:  # noqa: D401
        """The number of unique entities."""
        return self._num_entities

    @property
    def num_relations(self) -> int:  # noqa: D401
        """The number of unique relations."""
        return self._num_relations

    @property
    def num_triples(self) -> int:  # noqa: D401
        """The number of triples."""
        return self.mapped_triples.shape[0]

    @property
    def entity_id_to_label(self) -> Mapping[int, str]:  # noqa: D401
        """The mapping from entity IDs to their labels."""
        return invert_mapping(mapping=self.entity_to_id)

    @property
    def relation_id_to_label(self) -> Mapping[int, str]:  # noqa: D401
        """The mapping from relation IDs to their labels."""
        return invert_mapping(mapping=self.relation_to_id)

    def get_inverse_relation_id(self, relation: str) -> int:
        """Get the inverse relation identifier for the given relation."""
        if not self.create_inverse_triples:
            raise ValueError('Can not get inverse triple, they have not been created.')
        inverse_relation = self.relation_to_inverse[relation]
        return self.relation_to_id[inverse_relation]

    def __repr__(self):  # noqa: D105
        return f'{self.__class__.__name__}(path="{self.path}")'

    def create_slcwa_instances(self) -> SLCWAInstances:
        """Create sLCWA instances for this factory's triples."""
        return SLCWAInstances(
            mapped_triples=self.mapped_triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )

    def create_lcwa_instances(self, use_tqdm: Optional[bool] = None) -> LCWAInstances:
        """Create LCWA instances for this factory's triples."""
        s_p_to_multi_tails = _create_multi_label_tails_instance(
            mapped_triples=self.mapped_triples,
            use_tqdm=use_tqdm,
        )
        sp, multi_o = zip(*s_p_to_multi_tails.items())
        mapped_triples: torch.LongTensor = torch.tensor(sp, dtype=torch.long)
        labels = np.array([np.array(item) for item in multi_o], dtype=object)

        return LCWAInstances(
            mapped_triples=mapped_triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            labels=labels,
        )

    def map_triples_to_id(self, triples: Union[str, LabeledTriples]) -> MappedTriples:
        """Load triples and map to ids based on the existing id mappings of the triples factory.

        Works from either the path to a file containing triples given as string or a numpy array containing triples.
        """
        if isinstance(triples, str):
            triples = load_triples(triples)
        # Ensure 2d array in case only one triple was given
        triples = np.atleast_2d(triples)
        # FIXME this function is only ever used in tests
        return _map_triples_elements_to_ids(
            triples=triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )

    def split(
        self,
        ratios: Union[float, Sequence[float]] = 0.8,
        *,
        random_state: Union[None, int, np.random.RandomState] = None,
        randomize_cleanup: bool = False,
    ) -> List['TriplesFactory']:
        """Split a triples factory into a train/test.

        :param ratios: There are three options for this argument. First, a float can be given between 0 and 1.0,
         non-inclusive. The first triples factory will get this ratio and the second will get the rest. Second,
         a list of ratios can be given for which factory in which order should get what ratios as in ``[0.8, 0.1]``.
         The final ratio can be omitted because that can be calculated. Third, all ratios can be explicitly set in
         order such as in ``[0.8, 0.1, 0.1]`` where the sum of all ratios is 1.0.
        :param random_state: The random state used to shuffle and split the triples in this factory.
        :param randomize_cleanup: If true, uses the non-deterministic method for moving triples to the training set.
         This has the advantage that it doesn't necessarily have to move all of them, but it might be slower.

        .. code-block:: python

            ratio = 0.8  # makes a [0.8, 0.2] split
            training_factory, testing_factory = factory.split(ratio)

            ratios = [0.8, 0.1]  # makes a [0.8, 0.1, 0.1] split
            training_factory, testing_factory, validation_factory = factory.split(ratios)

            ratios = [0.8, 0.1, 0.1]  # also makes a [0.8, 0.1, 0.1] split
            training_factory, testing_factory, validation_factory = factory.split(ratios)
        """
        n_triples = self.triples.shape[0]

        # Prepare shuffle index
        idx = np.arange(n_triples)
        if random_state is None:
            random_state = random_non_negative_int()
            logger.warning(f'Using random_state={random_state} to split {self}')
        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        random_state.shuffle(idx)

        # Prepare split index
        if isinstance(ratios, float):
            ratios = [ratios]

        ratio_sum = sum(ratios)
        if ratio_sum == 1.0:
            ratios = ratios[:-1]  # vsplit doesn't take the final number into account.
        elif ratio_sum > 1.0:
            raise ValueError(f'ratios sum to more than 1.0: {ratios} (sum={ratio_sum})')

        sizes = [
            int(split_ratio * n_triples)
            for split_ratio in ratios
        ]
        # Take cumulative sum so the get separated properly
        split_idxs = np.cumsum(sizes)

        # Split triples
        triples_groups = np.vsplit(self.triples[idx], split_idxs)
        logger.info(f'split triples to groups of sizes {[triples.shape[0] for triples in triples_groups]}')

        # Make sure that the first element has all the right stuff in it
        triples_groups = _tf_cleanup_all(triples_groups, random_state=random_state if randomize_cleanup else None)

        # TODO: We do not really need to re-map the triples, but could perform the same split as done for the
        #       triples also for the mapped_triples

        for i, (triples, exp_size, exp_ratio) in enumerate(zip(triples_groups, sizes, ratios)):
            actual_size = triples.shape[0]
            actual_ratio = actual_size / exp_size * exp_ratio
            if actual_size != exp_size:
                logger.warning(
                    f'Requested ratio[{i}]={exp_ratio:.3f} (equal to size {exp_size}), but got {actual_ratio:.3f} '
                    f'(equal to size {actual_size}) to ensure that all entities/relations occur in train.',
                )

        # Make new triples factories for each group
        return [
            TriplesFactory.from_triples(
                triples=triples,
                entity_to_id=self.entity_to_id,
                relation_to_id=self.relation_to_id,
                compact_id=False,
            )
            for triples in triples_groups
        ]

    def get_most_frequent_relations(self, n: Union[int, float]) -> Set[str]:
        """Get the n most frequent relations.

        :param n: Either the (integer) number of top relations to keep or the (float) percentage of top relationships
         to keep
        """
        logger.info(f'applying cutoff of {n} to {self}')
        if isinstance(n, float):
            assert 0 < n < 1
            n = int(self.num_relations * n)
        elif not isinstance(n, int):
            raise TypeError('n must be either an integer or a float')

        counter = Counter(self.triples[:, 1])
        return {
            relation
            for relation, _ in counter.most_common(n)
        }

    def get_idx_for_entities(self, entities: Collection[str], invert: bool = False):
        """Get np.array indices for triples with the given entities."""
        entities = np.asanyarray(entities, dtype=self.triples.dtype)
        return (
            np.isin(self.triples[:, 0], entities, invert=invert)
            & np.isin(self.triples[:, 2], entities, invert=invert)
        )

    def get_idx_for_relations(self, relations: Collection[str], invert: bool = False):
        """Get np.array indices for triples with the given relations."""
        return np.isin(self.triples[:, 1], list(relations), invert=invert)

    def get_triples_for_relations(self, relations: Collection[str], invert: bool = False) -> LabeledTriples:
        """Get the labeled triples containing the given relations."""
        return self.triples[self.get_idx_for_relations(relations, invert=invert)]

    def new_with_relations(self, relations: Collection[str]) -> 'TriplesFactory':
        """Make a new triples factory only keeping the given relations."""
        idx = self.get_idx_for_relations(relations)
        logger.info(
            f'keeping {len(relations)}/{self.num_relations} relations'
            f' and {idx.sum()}/{self.num_triples} triples in {self}',
        )
        return TriplesFactory.from_triples(triples=self.triples[idx])

    def new_without_relations(self, relations: Collection[str]) -> 'TriplesFactory':
        """Make a new triples factory without the given relations."""
        idx = self.get_idx_for_relations(relations, invert=True)
        logger.info(
            f'removing {len(relations)}/{self.num_relations} relations'
            f' and {idx.sum()}/{self.num_triples} triples',
        )
        return TriplesFactory.from_triples(triples=self.triples[idx])

    def entity_word_cloud(self, top: Optional[int] = None):
        """Make a word cloud based on the frequency of occurrence of each entity in a Jupyter notebook.

        :param top: The number of top entities to show. Defaults to 100.

        .. warning::

            This function requires the ``word_cloud`` package. Use ``pip install pykeen[plotting]`` to
            install it automatically, or install it yourself with
            ``pip install git+https://github.com/kavgan/word_cloud.git``.
        """
        text = [f'{h} {t}' for h, _, t in self.triples]
        return self._word_cloud(text=text, top=top or 100)

    def relation_word_cloud(self, top: Optional[int] = None):
        """Make a word cloud based on the frequency of occurrence of each relation in a Jupyter notebook.

        :param top: The number of top relations to show. Defaults to 100.

        .. warning::

            This function requires the ``word_cloud`` package. Use ``pip install pykeen[plotting]`` to
            install it automatically, or install it yourself with
            ``pip install git+https://github.com/kavgan/word_cloud.git``.
        """
        text = [r for _, r, _ in self.triples]
        return self._word_cloud(text=text, top=top or 100)

    def _word_cloud(self, *, text: List[str], top: int):
        try:
            from word_cloud.word_cloud_generator import WordCloud
        except ImportError:
            logger.warning(
                'Could not import module `word_cloud`. '
                'Try installing it with `pip install git+https://github.com/kavgan/word_cloud.git`',
            )
            return

        from IPython.core.display import HTML
        word_cloud = WordCloud()
        return HTML(word_cloud.get_embed_code(text=text, topn=top))

    def tensor_to_df(
        self,
        tensor: torch.LongTensor,
        **kwargs: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> pd.DataFrame:
        """Take a tensor of triples and make a pandas dataframe with labels.

        :param tensor: shape: (n, 3)
            The triples, ID-based and in format (head_id, relation_id, tail_id).
        :param kwargs:
            Any additional number of columns. Each column needs to be of shape (n,). Reserved column names:
            {"head_id", "head_label", "relation_id", "relation_label", "tail_id", "tail_label"}.
        :return:
            A dataframe with n rows, and 6 + len(kwargs) columns.
        """
        # Input validation
        additional_columns = set(kwargs.keys())
        forbidden = additional_columns.intersection(TRIPLES_DF_COLUMNS)
        if len(forbidden) > 0:
            raise ValueError(
                f'The key-words for additional arguments must not be in {TRIPLES_DF_COLUMNS}, but {forbidden} were '
                f'used.',
            )

        # convert to numpy
        tensor = tensor.cpu().numpy()
        data = dict(zip(['head_id', 'relation_id', 'tail_id'], tensor.T))

        # vectorized label lookup
        entity_id_to_label = np.vectorize(self.entity_id_to_label.__getitem__)
        relation_id_to_label = np.vectorize(self.relation_id_to_label.__getitem__)
        for column, id_to_label in dict(
            head=entity_id_to_label,
            relation=relation_id_to_label,
            tail=entity_id_to_label,
        ).items():
            data[f'{column}_label'] = id_to_label(data[f'{column}_id'])

        # Additional columns
        for key, values in kwargs.items():
            # convert PyTorch tensors to numpy
            if torch.is_tensor(values):
                values = values.cpu().numpy()
            data[key] = values

        # convert to dataframe
        rv = pd.DataFrame(data=data)

        # Re-order columns
        columns = list(TRIPLES_DF_COLUMNS) + sorted(set(rv.columns).difference(TRIPLES_DF_COLUMNS))
        return rv.loc[:, columns]

    def new_with_restriction(
        self,
        entities: Optional[Collection[str]] = None,
        relations: Optional[Collection[str]] = None,
    ) -> 'TriplesFactory':
        """Make a new triples factory only keeping the given entities and relations, but keeping the ID mapping.

        :param entities:
            The entities of interest. If None, defaults to all entities.
        :param relations:
            The relations of interest. If None, defaults to all relations.

        :return:
            A new triples factory, which has only a subset of the triples containing the entities and relations of
            interest. The label-to-ID mapping is *not* modified.
        """
        if self.create_inverse_triples and relations is not None:
            logger.info(
                'Since %s already contain inverse relations, the relation filter is expanded to contain the inverse '
                'relations as well.',
                str(self),
            )
            relations = list(relations) + list(map(self.relation_to_inverse.__getitem__, relations))

        keep_mask = None

        # Filter for entities
        if entities is not None:
            keep_mask = self.get_idx_for_entities(entities=entities)
            logger.info('Keeping %d/%d entities', len(entities), self.num_entities)

        # Filter for relations
        if relations is not None:
            relation_mask = self.get_idx_for_relations(relations=relations)
            logger.info('Keeping %d/%d relations', len(relations), self.num_relations)
            keep_mask = relation_mask if keep_mask is None else keep_mask & relation_mask

        # No filtering happened
        if keep_mask is None:
            return self

        logger.info('Keeping %d/%d triples', keep_mask.sum(), self.num_triples)
        factory = TriplesFactory(
            triples=self.triples[keep_mask],
            create_inverse_triples=False,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            compact_id=False,
        )

        # manually copy the inverse relation mappings
        if self.create_inverse_triples:
            factory.create_inverse_triples = True
            factory.relation_to_inverse = self.relation_to_inverse
            factory._num_relations = self._num_relations

        return factory


def _tf_cleanup_all(
    triples_groups: List[np.ndarray],
    *,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> List[np.ndarray]:
    """Cleanup a list of triples array with respect to the first array."""
    reference, *others = triples_groups
    rv = []
    for other in others:
        if random_state is not None:
            reference, other = _tf_cleanup_randomized(reference, other, random_state)
        else:
            reference, other = _tf_cleanup_deterministic(reference, other)
        rv.append(other)
    return [reference, *rv]


def _tf_cleanup_deterministic(training: np.ndarray, testing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cleanup a triples array (testing) with respect to another (training)."""
    move_id_mask = _prepare_cleanup(training, testing)

    training = np.concatenate([training, testing[move_id_mask]])
    testing = testing[~move_id_mask]

    return training, testing


def _tf_cleanup_randomized(
    training: np.ndarray,
    testing: np.ndarray,
    random_state: Union[None, int, np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cleanup a triples array, but randomly select testing triples and recalculate to minimize moves.

    1. Calculate ``move_id_mask`` as in :func:`_tf_cleanup_deterministic`
    2. Choose a triple to move, recalculate move_id_mask
    3. Continue until move_id_mask has no true bits
    """
    if random_state is None:
        random_state = random_non_negative_int()
        logger.warning('Using random_state=%s', random_state)
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    move_id_mask = _prepare_cleanup(training, testing)

    # While there are still triples that should be moved to the training set
    while move_id_mask.any():
        # Pick a random triple to move over to the training triples
        idx = random_state.choice(move_id_mask.nonzero()[0])
        training = np.concatenate([training, testing[idx].reshape(1, -1)])

        # Recalculate the testing triples without that index
        testing_mask = np.ones_like(move_id_mask)
        testing_mask[idx] = False
        testing = testing[testing_mask]

        # Recalculate the training entities, testing entities, to_move, and move_id_mask
        move_id_mask = _prepare_cleanup(training, testing)

    return training, testing


def _prepare_cleanup(training: np.ndarray, testing: np.ndarray) -> np.ndarray:
    to_move_mask = None
    for col in [[0, 2], 1]:
        training_ids, test_ids = [np.unique(triples[:, col]) for triples in [training, testing]]
        to_move = test_ids[~np.isin(test_ids, training_ids)]
        this_to_move_mask = np.isin(testing[:, col], to_move)
        if this_to_move_mask.ndim > 1:
            this_to_move_mask = this_to_move_mask.any(axis=1)
        if to_move_mask is None:
            to_move_mask = this_to_move_mask
        else:
            to_move_mask = this_to_move_mask | to_move_mask

    return to_move_mask
