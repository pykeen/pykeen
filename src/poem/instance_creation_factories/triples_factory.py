# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

import logging
import os
import timeit
from collections import defaultdict
from typing import Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from .instances import CWAInstances, OWAInstances
from .utils import load_triples
from ..typing import EntityMapping, LabeledTriples, MappedTriples, RelationMapping
from ..utils import slice_triples

__all__ = [
    'TriplesFactory',
]

logger = logging.getLogger(__name__)


def _create_multi_label_objects_instance(mapped_triples: MappedTriples) -> Dict[Tuple[int, int], List[int]]:
    """Create for each (s,r) pair the multi object label."""
    logger.debug('Creating multi label objects instance')

    '''
    The mapped triples matrix has to be a numpy array to ensure correct pair hashing, as explained in
    https://github.com/mali-git/POEM_develop/commit/1bc71fe4eb2f24190425b0a4d0b9d6c7b9c4653a
    '''
    mapped_triples = mapped_triples.cpu().detach().numpy()

    s_r_to_multi_objects_new = _create_multi_label_instances(
        mapped_triples,
        element_1_index=0,
        element_2_index=1,
        label_index=2,
    )

    logger.debug('Created multi label objects instance')

    return s_r_to_multi_objects_new


def _create_multi_label_instances(
    mapped_triples: MappedTriples,
    element_1_index: int,
    element_2_index: int,
    label_index: int,
    use_tqdm: bool = True,
) -> Dict[Tuple[int, int], List[int]]:
    """Create for each (element_1, element_2) pair the multi-label."""
    instance_to_multi_label = defaultdict(set)

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


def _create_entity_mapping(triples: LabeledTriples) -> EntityMapping:
    """Create mapping from entity labels to IDs.

    :param triples: shape: (n, 3), dtype: str
    """
    # Split triples
    subjects, objects = triples[:, 0], triples[:, 2]
    # Sorting ensures consistent results when the triples are permuted
    entity_labels = sorted(set(subjects).union(objects))
    # Create mapping
    entity_label_to_id = {label: i for (i, label) in enumerate(entity_labels)}
    return entity_label_to_id


def _create_relation_mapping(triples: LabeledTriples) -> RelationMapping:
    """Create mapping from relation labels to IDs.

    :param triples: shape: (n, 3), dtype: str
    """
    # Extract relation labels
    relations = triples[:, 1]
    # Sorting ensures consistent results when the triples are permuted
    relation_labels = sorted(set(relations))
    # Create mapping
    relation_label_to_id = {label: i for (i, label) in enumerate(relation_labels)}
    return relation_label_to_id


def _map_triples_elements_to_ids(
    triples: LabeledTriples,
    entity_to_id: EntityMapping,
    relation_to_id: RelationMapping,
) -> MappedTriples:
    """Map entities and relations to pre-defined ids."""
    heads, relations, tails = slice_triples(triples)

    # When triples that don't exist are trying to be mapped, they get the id "-1"
    subject_column = np.vectorize(entity_to_id.get)(heads, [-1])
    relation_column = np.vectorize(relation_to_id.get)(relations, [-1])
    object_column = np.vectorize(entity_to_id.get)(tails, [-1])

    # Filter all non-existent triples
    subject_filter = subject_column < 0
    relation_filter = relation_column < 0
    object_filter = object_column < 0
    num_no_subject = subject_filter.sum()
    num_no_relation = relation_filter.sum()
    num_no_object = object_filter.sum()

    if (num_no_subject > 0) or (num_no_relation > 0) or (num_no_object > 0):
        logger.warning(
            "You're trying to map triples with entities and/or relations that are not in the training set."
            "These triples will be excluded from the mapping")
        non_mappable_triples = (subject_filter | relation_filter | object_filter)
        subject_column = subject_column[~non_mappable_triples, None]
        relation_column = relation_column[~non_mappable_triples, None]
        object_column = object_column[~non_mappable_triples, None]
        logger.warning(
            f"In total {non_mappable_triples.sum():.0f} from {triples.shape[0]:.0f} triples were filtered out")

    triples_of_ids = np.concatenate([subject_column, relation_column, object_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.long)
    # Note: Unique changes the order of the triples
    # Note: Using unique means implicit balancing of training samples
    unique_mapped_triples = np.unique(ar=triples_of_ids, axis=0)
    return torch.tensor(unique_mapped_triples, dtype=torch.long)


class TriplesFactory:
    """Create instances given the path to triples."""

    #: The integer indexes for each entity
    all_entities: torch.LongTensor

    #: The mapping from entities' labels to their indexes
    entity_to_id: EntityMapping

    #: The integer indexes for each relation
    all_relations: torch.LongTensor

    #: The mapping from relations' labels to their indexes
    relation_to_id: RelationMapping

    #: A three-column matrix where each row are the subject label,
    #: relation label, then object label
    triples: LabeledTriples

    #: A three-column matrix where each row are the subject identifier,
    #: relation identifier, then object identifier
    mapped_triples: MappedTriples

    def __init__(
        self,
        *,
        path: Union[None, str, TextIO] = None,
        triples: Optional[LabeledTriples] = None,
        create_inverse_triples: bool = False,
        entity_to_id: EntityMapping = None,
        relation_to_id: RelationMapping = None,
    ) -> None:
        """Initialize the triples factory.

        :param path: The path to a 3-column TSV file with triples in it. If not specified,
         you should specify ``triples``.
        :param triples:  A 3-column numpy array with triples in it. If not specified,
         you should specify ``path``
        :param create_inverse_triples: Should inverse triples be created? Defaults to False.
        """
        if path is None and triples is None:
            raise ValueError('Must specify either triples or path')
        elif path is not None and triples is not None:
            raise ValueError('Must not specify both triples and path')
        elif path is not None:
            if isinstance(path, str):
                self.path = os.path.abspath(path)
            elif isinstance(path, TextIO):
                self.path = os.path.abspath(path.name)
            else:
                raise TypeError(f'path is invalid type: {type(path)}')

            # TODO: Check if lazy evaluation would make sense
            self.triples = load_triples(path)
        else:  # triples is not None
            self.path = '<None>'
            self.triples = triples

        # Generate entity mapping if necessary
        if entity_to_id is None:
            entity_to_id = _create_entity_mapping(triples=self.triples)
        self.entity_to_id = entity_to_id

        # Generate relation mapping if necessary
        if relation_to_id is None:
            relation_to_id = _create_relation_mapping(triples=self.triples)
        self.relation_to_id = relation_to_id

        # Store entity and relation IDs
        self.all_entities = torch.arange(self.num_entities)
        self.all_relations = torch.arange(self.num_relations)

        # Map triples of labels to triples of IDs.
        self.mapped_triples = _map_triples_elements_to_ids(
            triples=self.triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )

        self.create_inverse_triples = create_inverse_triples

        # This creates inverse triples and appends them to the mapped triples
        if self.create_inverse_triples:
            self._create_inverse_triples()

    @property
    def num_entities(self) -> int:  # noqa: D401
        """The number of unique entities."""
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:  # noqa: D401
        """The number of unique relations."""
        return len(self.relation_to_id)

    def get_inverse_relation_id(self, relation: str) -> int:
        """Get the inverse relation identifier for the given relation."""
        if not self.create_inverse_triples:
            raise ValueError('Can not get inverse triple, they have not been created.')
        return self.relation_to_id[relation] + self.num_relations // 2

    def __repr__(self):  # noqa: D105
        return f'{self.__class__.__name__}(path="{self.path}")'

    def _create_inverse_triples(self) -> None:
        start = timeit.default_timer()
        logger.debug('Creating inverse triples')
        inverse_triples = self.mapped_triples.clone().flip(1)
        inverse_triples[:, 1:2] += self.num_relations
        self.mapped_triples = torch.cat((self.mapped_triples, inverse_triples), dim=0)
        logger.debug(f'Creating inverse triples done after {timeit.default_timer() - start:.2f} seconds')
        # The newly added inverse relations have to be added to the relation_id mapping
        inverse_triples_to_id_mapping = {
            f"{relation}_inverse": relation_id + self.num_relations
            for relation, relation_id in self.relation_to_id.items()
        }
        self.relation_to_id.update(inverse_triples_to_id_mapping)

    def create_owa_instances(self) -> OWAInstances:
        """Create OWA instances for this factory's triples."""
        return OWAInstances(
            mapped_triples=self.mapped_triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )

    def create_cwa_instances(self) -> CWAInstances:
        """Create CWA instances for this factory's triples."""
        s_r_to_multi_objects = _create_multi_label_objects_instance(
            mapped_triples=self.mapped_triples,
        )
        sr, multi_o = zip(*s_r_to_multi_objects.items())
        mapped_triples: torch.LongTensor = torch.tensor(sr, dtype=torch.long)
        labels = np.array(multi_o)

        return CWAInstances(
            mapped_triples=mapped_triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            labels=labels,
        )
